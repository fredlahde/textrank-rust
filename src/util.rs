use std::cmp::Ordering;
#[allow(clippy::comparison_chain)]
pub(crate) fn cmp_f64(a: f64, b: f64) -> Ordering {
    if a.is_nan() {
        return Ordering::Less;
    }
    if b.is_nan() {
        return Ordering::Greater;
    }
    if a < b {
        return Ordering::Greater;
    } else if a > b {
        return Ordering::Less;
    }
    Ordering::Equal
}
