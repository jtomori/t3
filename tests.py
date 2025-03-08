import unittest

from t3 import vad


PURE_VOICE = [4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # Voice without any background sounds
NO_VOICE = [3, 8, 20, 28, 35, 37, 52, 57, 67, 69, 89, 92]
VOICE_BG_SOUNDS = [5, 39, 54, 139, 186]  # Voice with background sounds
SONGS = [0, 1, 2, 83, 206]


class VAD(unittest.TestCase):
    def test_pure_voice(self):
        for id in PURE_VOICE:
            with self.subTest(id):
                self.assertTrue(vad.detect_speech(f"ogg/Mein Woerter-Bilderbuch Unser Zuhause_{id}.ogg"))

    def test_no_voice(self):
        for id in NO_VOICE:
            with self.subTest(id):
                self.assertFalse(vad.detect_speech(f"ogg/Mein Woerter-Bilderbuch Unser Zuhause_{id}.ogg"))

    def test_voice_bg_sounds(self):
        for id in VOICE_BG_SOUNDS:
            with self.subTest(id):
                self.assertTrue(vad.detect_speech(f"ogg/Mein Woerter-Bilderbuch Unser Zuhause_{id}.ogg"))

    def test_songs(self):
        for id in SONGS:
            with self.subTest(id):
                self.assertTrue(vad.detect_speech(f"ogg/Mein Woerter-Bilderbuch Unser Zuhause_{id}.ogg"))


if __name__ == "__main__":
    unittest.main(verbosity=3)
