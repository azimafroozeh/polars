use std::fmt;
use std::str::FromStr;

/// Bloom-filter row-group pruning mode: disabled, or enabled with a filter-read strategy.
#[repr(u8)]
#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
pub enum BloomPruning {
    /// Pruning disabled.
    Off = 0,
    /// Pruning on local sources with block reads; cloud sources are not probed (serial
    /// per-row-group probes on high-latency storage cost more than they save).
    Auto = 1,
    /// Pruning enabled; always fetch the whole filter in one request.
    Whole = 2,
    /// Pruning enabled; always fetch the header, then only the block(s) the hashes map to.
    Blocks = 3,
}

impl fmt::Display for BloomPruning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_static_str())
    }
}

impl FromStr for BloomPruning {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" | "false" | "off" => Ok(Self::Off),
            "1" | "true" | "auto" => Ok(Self::Auto),
            "whole" => Ok(Self::Whole),
            "blocks" => Ok(Self::Blocks),
            v => Err(format!(
                "`bloom_filter_prune` must be one of {{'0', 'false', 'off', '1', 'true', \
                 'auto', 'whole', 'blocks'}}, got {v}",
            )),
        }
    }
}

impl BloomPruning {
    pub(crate) fn from_discriminant(d: u8) -> Self {
        match d {
            0 => Self::Off,
            1 => Self::Auto,
            2 => Self::Whole,
            3 => Self::Blocks,
            _ => unreachable!(),
        }
    }

    pub fn as_static_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::Whole => "whole",
            Self::Blocks => "blocks",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_and_round_trip() {
        for (s, expected) in [
            ("0", BloomPruning::Off),
            ("false", BloomPruning::Off),
            ("off", BloomPruning::Off),
            ("1", BloomPruning::Auto),
            ("true", BloomPruning::Auto),
            ("auto", BloomPruning::Auto),
            ("whole", BloomPruning::Whole),
            ("blocks", BloomPruning::Blocks),
        ] {
            assert_eq!(s.parse::<BloomPruning>().unwrap(), expected);
        }
        for mode in [
            BloomPruning::Off,
            BloomPruning::Auto,
            BloomPruning::Whole,
            BloomPruning::Blocks,
        ] {
            assert_eq!(mode.as_static_str().parse::<BloomPruning>().unwrap(), mode);
            assert_eq!(BloomPruning::from_discriminant(mode as u8), mode);
        }
        assert!("on".parse::<BloomPruning>().is_err());
        assert!("AUTO".parse::<BloomPruning>().is_err());
        assert!("".parse::<BloomPruning>().is_err());
    }
}
