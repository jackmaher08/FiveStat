import unittest
from fpl_live import extract_live_minutes


class FPLLiveTests(unittest.TestCase):
    def test_list(self):
        self.assertEqual(extract_live_minutes([{"id": 10, "stats": {"minutes": 90}}]), {10: 90})

    def test_dictionary(self):
        self.assertEqual(extract_live_minutes({"10": {"stats": {"minutes": 0}}}), {10: 0})


if __name__ == "__main__":
    unittest.main()
