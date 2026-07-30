//! Row-group pruning using Parquet split-block bloom filters.
//!
//! Called from [`super::statistics::calculate_row_group_pred_pushdown_skip_mask`].
//!
//! Statistics (min/max) run first; blooms are probed only on row groups not already
//! skipped by stats. The bloom mask is OR-merged with the statistics mask (set bit = skip).
//!
//! Future dictionary-based skipping: see the design note on
//! [`super::statistics::calculate_row_group_pred_pushdown_skip_mask`].

use std::ops::Range;
use std::sync::Arc;

use arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_core::prelude::{DataType, PlHashMap, Scalar};
use polars_error::PolarsResult;
use polars_io::predicates::{
    ScanIOPredicate, SpecializedColumnPredicate, bloom_hashes_for_scalars,
};
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_parquet::parquet::bloom_filter::{
    BLOCK_SIZE, any_hashes_might_be_in_blocks, bloom_filter_layout, might_contain_any_hashes,
    prefer_block_reads, unique_block_indices,
};
use polars_parquet::read::{ColumnChunkMetadata, RowGroupMetadata};
use polars_utils::pl_str::PlSmallStr;

use super::projection::ArrowFieldProjection;

/// On-disk Arrow field name and precomputed hashes used to probe that column's Bloom filter.
pub(super) struct BloomColumnPred {
    pub(super) arrow_field_name: PlSmallStr,
    pub(super) hashes: Box<[u64]>,
}

/// Combine statistics- and Bloom filter-derived skip masks.
///
/// Either path may mark a row group for skipping; we take the union (`|`).
pub(super) fn merge_row_group_skip_masks(
    statistics_mask: Bitmap,
    bloom_mask: Option<Bitmap>,
) -> Bitmap {
    let Some(bloom_mask) = bloom_mask else {
        return statistics_mask;
    };
    debug_assert_eq!(statistics_mask.len(), bloom_mask.len());
    &statistics_mask | &bloom_mask
}

/// Collect Bloom filter-eligible columns and hash literals once per file.
pub(super) fn collect_bloom_preds(
    predicate: &ScanIOPredicate,
    projected_arrow_fields: &[ArrowFieldProjection],
) -> Option<Vec<BloomColumnPred>> {
    // Output name -> on-disk Arrow name (bloom offsets are keyed by Arrow name in chunk metadata).
    // Casting projections are excluded: the literal is hashed in the output dtype while the file's
    // bloom was hashed in the file dtype, so probing would wrongly skip row groups.
    let output_to_arrow_name: PlHashMap<_, _> = projected_arrow_fields
        .iter()
        .filter(|p| DataType::from_arrow_field(p.arrow_field()) == *p.output_dtype())
        .map(|p| (p.output_name().clone(), p.arrow_field().name.clone()))
        .collect();

    let mut bloom_preds = Vec::new();
    for (output_name, (_, specialized)) in predicate.column_predicates.predicates.iter() {
        if !predicate.live_columns.contains(output_name) {
            continue;
        }
        let Some(values) = bloom_pred_values(specialized) else {
            continue;
        };
        let Some(hashes) = bloom_hashes_for_scalars(values) else {
            continue;
        };
        let Some(arrow_field_name) = output_to_arrow_name.get(output_name) else {
            continue;
        };
        bloom_preds.push(BloomColumnPred {
            arrow_field_name: arrow_field_name.clone(),
            hashes,
        });
    }

    (!bloom_preds.is_empty()).then_some(bloom_preds)
}

/// For each row group not already skipped by `statistics_mask`, probe on-disk Bloom filters.
///
/// Returns `None` if there are no Bloom predicates. Otherwise returns a bitmap of length
/// `row_groups.len()` where `true` means skip (value cannot be in Bloom) for row groups not
/// already skipped by statistics; for those skipped by statistics, the Bloom mask is irrelevant.
pub(super) async fn bloom_filter_row_group_skip_mask(
    row_groups: &[RowGroupMetadata],
    byte_source: Arc<DynByteSource>,
    bloom_preds: Option<Vec<BloomColumnPred>>,
    statistics_mask: &Bitmap,
    whole_read: bool,
) -> PolarsResult<Option<Bitmap>> {
    let Some(bloom_preds) = bloom_preds.as_deref().filter(|p| !p.is_empty()) else {
        return Ok(None);
    };

    debug_assert_eq!(statistics_mask.len(), row_groups.len());

    let mut skip = BitmapBuilder::with_capacity(row_groups.len());
    let mut bitset = Vec::new();

    for (i, rg) in row_groups.iter().enumerate() {
        if statistics_mask.get_bit(i) {
            // Already skipped by min/max; the bloom probe cannot change the merged mask.
            skip.push(false);
            continue;
        }
        skip.push(
            should_skip_row_group(rg, bloom_preds, &byte_source, &mut bitset, whole_read).await?,
        );
    }

    Ok(Some(skip.freeze()))
}

/// Literals to hash into the Bloom filter; `None` for non-point predicates (ranges, strings, …).
fn bloom_pred_values(specialized: &Option<SpecializedColumnPredicate>) -> Option<&[Scalar]> {
    use SpecializedColumnPredicate as S;
    match specialized {
        Some(S::Equal(s)) => Some(std::slice::from_ref(s)),
        Some(S::EqualOneOf(v)) => {
            (v.len() <= polars_config::config().bloom_in_filter_threshold()).then_some(v.as_ref())
        },
        // Ranges and substring/regex predicates cannot be answered by a point-membership bloom.
        Some(S::Between(..) | S::StartsWith(_) | S::EndsWith(_) | S::RegexMatch(_)) | None => None,
    }
}

/// Byte range of the serialized Bloom filter for a column chunk, if present and valid.
///
/// Returns `None` on missing or unusable metadata (caller treats as "might contain").
fn bloom_byte_range(meta: &ColumnChunkMetadata) -> Option<Range<usize>> {
    let offset = usize::try_from(meta.bloom_filter_offset()?).ok()?;
    let len = meta.bloom_filter_length()?;
    if len <= 0 {
        return None;
    }
    let end = offset.checked_add(len as usize)?;
    Some(offset..end)
}

/// Returns `true` if Blooms prove this row group cannot satisfy the filter conjuncts.
async fn should_skip_row_group(
    rg: &RowGroupMetadata,
    bloom_preds: &[BloomColumnPred],
    byte_source: &DynByteSource,
    bitset: &mut Vec<u8>,
    whole_read: bool,
) -> PolarsResult<bool> {
    for pred in bloom_preds {
        let Some(idxs) = rg.columns_idxs_under_root_iter(pred.arrow_field_name.as_str()) else {
            continue;
        };
        // Nested fields map to 0 or 2+ chunks; blooms are per chunk, so none applies.
        if idxs.len() != 1 {
            continue;
        }
        let column_metadata = &rg.parquet_columns()[idxs[0]];
        let Some(range) = bloom_byte_range(column_metadata) else {
            continue;
        };

        // An `Err` is a genuine byte-source IO failure, so propagate.
        let any_might_match =
            probe_bloom_hashes(&pred.hashes, range, byte_source, bitset, whole_read).await?;

        if !any_might_match {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Max bytes to read for the Thrift Bloom filter header (~12-20 bytes; 64 leaves margin).
const BLOOM_HEADER_READ_CAP: usize = 64;

/// Max filter size fetched as one request under whole-filter reads; one round trip beats the
/// dependent header-then-blocks pair until transfer time exceeds a round trip, which at
/// object-store bandwidth sits well above realistic filter sizes (~1 MiB writer caps).
const BLOOM_WHOLE_READ_CAP: usize = 4 * 1024 * 1024;

/// Probe Bloom filter literals.
///
/// With `whole_read` (and up to [`BLOOM_WHOLE_READ_CAP`]) the filter is fetched in a single
/// request; otherwise the header is fetched first and then only the needed split-block(s) when
/// cheaper than the full slice. Corrupt, truncated, or unsupported filters resolve to `Ok(true)`
/// (inconclusive, may contain matches); `Err` is returned only for a byte-source IO failure.
async fn probe_bloom_hashes(
    hashes: &[u64],
    bloom_range: Range<usize>,
    byte_source: &DynByteSource,
    bitset: &mut Vec<u8>,
    whole_read: bool,
) -> PolarsResult<bool> {
    if whole_read && bloom_range.len() <= BLOOM_WHOLE_READ_CAP {
        let bloom_bytes = byte_source.get_range(bloom_range).await?;
        // Corrupt or truncated bloom bytes: inconclusive, do not skip the row group.
        return Ok(might_contain_any_hashes(bloom_bytes.as_ref(), hashes, bitset).unwrap_or(true));
    }

    let header_end = bloom_range
        .end
        .min(bloom_range.start.saturating_add(BLOOM_HEADER_READ_CAP));
    if header_end <= bloom_range.start {
        return Ok(true);
    }

    let prefix = byte_source.get_range(bloom_range.start..header_end).await?;
    // Unsupported, truncated, or corrupt header: treat as inconclusive (may contain matches).
    let Some(layout) = bloom_filter_layout(prefix.as_ref()).ok().flatten() else {
        return Ok(true);
    };

    let bitset_start = bloom_range.start + layout.header_len;
    let bitset_end = bitset_start
        .checked_add(layout.bitset_num_bytes)
        .filter(|&end| end <= bloom_range.end);
    if bitset_end.is_none() {
        return Ok(true);
    }

    let block_indices = unique_block_indices(hashes, layout.bitset_num_bytes);
    if !prefer_block_reads(block_indices.len(), &layout, bloom_range.len()) {
        // The prefix already holds the whole filter when the range fits inside the header cap.
        let bloom_bytes = if header_end == bloom_range.end {
            prefix
        } else {
            byte_source.get_range(bloom_range).await?
        };
        // Corrupt or truncated bloom bytes: inconclusive, do not skip the row group.
        return Ok(might_contain_any_hashes(bloom_bytes.as_ref(), hashes, bitset).unwrap_or(true));
    }

    // Shared so the range request and the `blocks_by_offset` lookup derive offsets identically.
    let block_start = |idx: usize| bitset_start + idx * BLOCK_SIZE;

    let mut block_ranges: Vec<Range<usize>> = block_indices
        .iter()
        .map(|&idx| block_start(idx)..block_start(idx) + BLOCK_SIZE)
        .collect();
    let blocks_by_offset = byte_source.get_ranges(&mut block_ranges).await?;

    Ok(any_hashes_might_be_in_blocks(
        hashes,
        layout.bitset_num_bytes,
        |idx| blocks_by_offset.get(&block_start(idx)).map(|b| b.as_ref()),
    ))
}
