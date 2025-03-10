# TipToi Translation (t3)

## Usage
- Once you've followed the setup instructions below you can run the application
- Run the following from repository's root
  - `python -m t3 'Mein Woerter-Bilderbuch Unser Zuhause.gme' workdir`
  - This will translate `Mein Woerter-Bilderbuch Unser Zuhause.gme` and store the translated GME and intermediate files in the `workdir`
  - Run `python -m t3 -h` to see all available options

## Development

### Setup
- `sudo apt install sox ffmpeg`
- `pip install numpy typing_extensions`
- `pip install -r requirements.txt`
- Store [SeamlessExpressive](https://huggingface.co/facebook/seamless-expressive) models in the `SeamlessExpressive` folder in repository's root
- Download https://github.com/entropia/tip-toi-reveng/releases/tag/1.11 and make sure `tttool` is in your PATH. E.g. with `source env.sh`.

### Tests & code checks
- `python tests.py`
- `./checks.sh`
