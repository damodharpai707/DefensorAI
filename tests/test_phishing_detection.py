import unittest
from src.detection.phishing_detection_algorithm import detect_phishing

class TestPhishingDetectionAlgorithm(unittest.TestCase):
    def test_domain_detection_phishing(self):
        """Test rule-based detection for phishing domains."""
        domain = "secure-login.bank.com"
        result = detect_phishing(domain, "domain")
        self.assertEqual(result["status"], "Phishing")
        self.assertGreaterEqual(result["score"], 0.5, "Phishing score is below threshold.")

    def test_domain_detection_legitimate(self):
        """Test rule-based detection for legitimate domains."""
        domain = "example.com"
        result = detect_phishing(domain, "domain")
        self.assertEqual(result["status"], "Legitimate")
        self.assertLess(result["score"], 0.5, "Legitimate score is above threshold.")

    def test_url_detection_phishing(self):
        """Test rule-based detection for phishing URLs."""
        url = "http://secure-login.bank.com/login"
        result = detect_phishing(url, "url")
        self.assertEqual(result["status"], "Phishing")
        self.assertGreaterEqual(result["score"], 0.5, "Phishing score is below threshold.")

    def test_url_detection_legitimate(self):
        """Test rule-based detection for legitimate URLs."""
        url = "https://safe-site.example.com"
        result = detect_phishing(url, "url")
        self.assertEqual(result["status"], "Legitimate")
        self.assertLess(result["score"], 0.5, "Legitimate score is above threshold.")

    def test_invalid_entry_type(self):
        """Test handling of invalid entry type."""
        with self.assertRaises(ValueError):
            detect_phishing("example.com", "invalid_type")

    def test_edge_case_empty_domain(self):
        """Test edge case with empty domain."""
        domain = ""
        result = detect_phishing(domain, "domain")
        self.assertEqual(result["status"], "Legitimate")
        self.assertEqual(result["score"], 0.0, "Empty domain should have zero score.")

    def test_edge_case_empty_url(self):
        """Test edge case with empty URL."""
        url = ""
        result = detect_phishing(url, "url")
        self.assertEqual(result["status"], "Legitimate")
        self.assertEqual(result["score"], 0.0, "Empty URL should have zero score.")

    def test_suspicious_keywords_detection(self):
        """Test detection of suspicious keywords in domains and URLs."""
        domain = "verify-account.example.com"
        result = detect_phishing(domain, "domain")
        self.assertEqual(result["status"], "Phishing")
        self.assertIn("has_suspicious_keywords", result["features"])
        self.assertTrue(result["features"]["has_suspicious_keywords"])

        url = "https://secure-bank.com/login"
        result = detect_phishing(url, "url")
        self.assertEqual(result["status"], "Phishing")
        self.assertIn("has_suspicious_keywords", result["features"])
        self.assertTrue(result["features"]["has_suspicious_keywords"])

if __name__ == "__main__":
    unittest.main()
