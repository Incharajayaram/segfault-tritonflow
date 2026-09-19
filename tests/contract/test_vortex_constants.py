"""Contract test for Task E5: Vortex Upstream Hardware Constants Verification."""

from tools.check_vortex_constants import verify_constants


def test_vortex_constants_match_upstream():
    ok, errors = verify_constants()
    assert ok, f"Vortex hardware constants drift detected: {errors}"
