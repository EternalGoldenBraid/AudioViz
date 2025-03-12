(assets/demo.mp4)

# To find out the correct audio devices run `tests/test_read_live_audio.py`:
```
  0 HDA NVidia: SAMSUNG (hw:0,3), ALSA (0 in, 8 out)
  ...
  8 Scarlett Solo USB: Audio (hw:3,0), ALSA (2 in, 2 out)
  9 hdmi, ALSA (0 in, 8 out)
```

and create the appropriate enum:
```
class AudioDevicesSomeSystem(Enum):
    SAMSUNG = 0
    SCARLETT = 8
```
