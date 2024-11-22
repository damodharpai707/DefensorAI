import unittest
from src.detection.phishing_detection_algorithm import detect_phishing

class TestPhishingDetectionAlgorithm(unittest.TestCase):
    def test_domain_detection(self):
        """Test rule-based detection for domains."""
        domain = "secure-login.bank.com"
        result = detect_phishing(domain, "domain")
        self.assertEqual(result, "Phishing")

        domain = "example.com"
        result = detect_phishing(domain, "domain")
        self.assertEqual(result, "Legitimate")

    def test_url_detection(self):
        """Test rule-based detection for URLs."""
        url = "http://secure-login.bank.com"
        result = detect_phishing(url, "url")
        self.assertEqual(result, "Phishing")

        url = "https://safe-site.example.com"
        result = detect_phishing(url, "url")
        self.assertEqual(result, "Legitimate")

if __name__ == "__main__":
    unittest.main()
