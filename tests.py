import os
import unittest

from t3 import audio_utils, s2st


PURE_VOICE = [4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # Voice without any background sounds
NO_VOICE = [3, 8, 20, 28, 35, 37, 52, 57, 67, 69, 89, 92]
VOICE_BG_SOUNDS = [5, 39, 54, 139, 186]  # Voice with background sounds
SONGS = [0, 1, 2, 83, 206]


class VAD(unittest.TestCase):
    def test_pure_voice(self):
        for id in PURE_VOICE:
            with self.subTest(id):
                self.assertTrue(audio_utils.detect_speech(f"ogg/Mein Woerter-Bilderbuch Unser Zuhause_{id}.ogg"))

    def test_no_voice(self):
        for id in NO_VOICE:
            with self.subTest(id):
                self.assertFalse(audio_utils.detect_speech(f"ogg/Mein Woerter-Bilderbuch Unser Zuhause_{id}.ogg"))

    def test_voice_bg_sounds(self):
        for id in VOICE_BG_SOUNDS:
            with self.subTest(id):
                self.assertTrue(audio_utils.detect_speech(f"ogg/Mein Woerter-Bilderbuch Unser Zuhause_{id}.ogg"))

    def test_songs(self):
        for id in SONGS:
            with self.subTest(id):
                self.assertTrue(audio_utils.detect_speech(f"ogg/Mein Woerter-Bilderbuch Unser Zuhause_{id}.ogg"))


class Various(unittest.TestCase):
    def test_too_long_1(self):
        self.assertFalse(audio_utils.check_audio_length("ogg/Mein Woerter-Bilderbuch Unser Zuhause_0.ogg"))

    def test_too_long_2(self):
        self.assertTrue(audio_utils.check_audio_length("ogg/Mein Woerter-Bilderbuch Unser Zuhause_4.ogg"))


class S2ST(unittest.TestCase):
    def test_inference(self):
        out = s2st.translate_audio_files(
            ["ogg/Mein Woerter-Bilderbuch Unser Zuhause_4.ogg", "ogg/Mein Woerter-Bilderbuch Unser Zuhause_9.ogg"],
            "tmp",
            force_cpu=True
        )

        path_1 = out[0].path
        text_1 = out[0].text
        path_2 = out[1].path
        text_2 = out[1].text

        self.assertTrue(os.path.exists(path_1))
        self.assertEqual(path_1, "tmp/Mein Woerter-Bilderbuch Unser Zuhause_4.mp3")
        self.assertEqual(text_1, "Welcome to the Hoffmann family. Lara and Papa are just returning from shopping. The two of them went to the market. They bought fresh fruit and vegetables. Rocky, the family's dog, runs up to them cheerfully barking. At the front door, Mama is standing and talking to Aunt Julia. Lara's brothers, Finn and David, are also there.")

        self.assertTrue(os.path.exists(path_2))
        self.assertEqual(path_2, "tmp/Mein Woerter-Bilderbuch Unser Zuhause_9.mp3")


if __name__ == "__main__":
    unittest.main(verbosity=3)
