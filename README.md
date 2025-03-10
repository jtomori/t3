# TipToi Translation (t3)

## Usage
- Once you've followed the setup instructions below you can run the application
- Run the following from repository's root
  - `python -m t3 'Rekorde im Tierreich.gme' workdir`
  - This will translate `Rekorde im Tierreich.gme` and store the translated GME and intermediate files in the `workdir`
  - Run `python -m t3 -h` to see all available options

## Development

### Setup
- `sudo apt install sox ffmpeg`
- `pip install numpy typing_extensions`
- `pip install -r requirements.txt`
- Store [SeamlessExpressive](https://huggingface.co/facebook/seamless-expressive) models in the `SeamlessExpressive` folder in repository's root
- Compile `libtiptoi.c`: `gcc tip-toi-reveng/libtiptoi.c -o libtiptoi`

### Tests & code checks
- `python tests.py`
- `./checks.sh`
