import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import server


class TestServer(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(server.app)

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

    def test_home_serves_ui(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<html", response.text)

    def test_phonemes(self):
        response = self.client.post("/phonemes", data={"text": "hello world"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertGreater(len(body["phonemes"]), 0)
        self.assertEqual(len(body["phonemes"]), len(body["words"]))

    @patch("server.speech.transcribe", return_value="HELLO")
    def test_speech2text(self, _):
        import io
        import numpy as np
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, np.zeros(16000, dtype="float32"), 16000, format="WAV")
        buf.seek(0)
        response = self.client.post("/speech2text", files={"file": ("rec.wav", buf, "audio/wav")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"transcript": "HELLO"})

    def test_ui_assets_and_languages(self):
        for path in ("/static/ui.js", "/static/audio.js", "/static/viseme.js", "/static/assets/logo.svg"):
            self.assertEqual(self.client.get(path).status_code, 200, path)
        home = self.client.get("/").text
        for element in ("record-btn", "language-select", "expected-text", "word-chips", "score-ring"):
            self.assertIn(f'id="{element}"', home)
        languages = self.client.get("/languages").json()
        self.assertEqual(languages["default"], "en")
        self.assertIn({"code": "en", "name": "English"}, languages["languages"])


class TestServerDeployment(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(server.app)

    def test_ready_follows_model_loading(self):
        server._ready.clear()
        self.assertEqual(self.client.get("/ready").status_code, 503)
        server._ready.set()
        self.assertEqual(self.client.get("/ready").json(), {"status": "ready"})

    def test_api_key_required_when_configured(self):
        with patch.object(server, "API_KEY", "secret"):
            data = {"text": "hello"}
            self.assertEqual(self.client.post("/phonemes", data=data).status_code, 401)
            self.assertEqual(self.client.post("/phonemes", data=data, headers={"X-API-Key": "nope"}).status_code, 401)
            self.assertEqual(self.client.post("/phonemes", data=data, headers={"X-API-Key": "secret"}).status_code, 200)
            # UI and probes stay public
            self.assertEqual(self.client.get("/").status_code, 200)
            self.assertEqual(self.client.get("/health").status_code, 200)

    def test_only_enabled_languages(self):
        languages = self.client.get("/languages").json()
        self.assertEqual([l["code"] for l in languages["languages"]], ["en"])
        self.assertEqual(self.client.post("/phonemes", data={"text": "bonjour", "lang": "fr"}).status_code, 422)

    def test_upload_and_text_limits(self):
        import io
        with patch.object(server, "MAX_UPLOAD_BYTES", 10):
            response = self.client.post("/speech2text", files={"file": ("a.wav", io.BytesIO(b"x" * 100), "audio/wav")})
            self.assertEqual(response.status_code, 413)
        with patch.object(server, "MAX_TEXT_CHARS", 5):
            self.assertEqual(self.client.post("/phonemes", data={"text": "too long text"}).status_code, 422)
