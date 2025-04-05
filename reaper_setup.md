# Some notes

## Scarlet solo

- For both guitar and mic make sure Reaper audio device is set to PulseAudio (Laptop as PipeWire)
- In pavucontrol configuration tab set Scarlet Solo profile to `Pro Audio`

The mic and guitar should now be available in mono channels 1 and 2 respectively.


## MIDI keyboard
Is it listed from:
```
lsusb
>> Bus 001 Device 123: ID 0763:0199 M-Audio Axiom 49
```

Recognised as a MIDI device?:
```
aconnect -l
```

Check if signals are coming through:
```
aseqdump -l
>> 
Port    Client name                      Port name
  0:0    System                           Timer
  0:1    System                           Announce
 14:0    Midi Through                     Midi Through Port-0
 16:0    USB Axiom 49                     USB Axiom 49 MIDI 1
 16:1    USB Axiom 49                     USB Axiom 49 MIDI 2
```

Listen on port 16 and press notes:
```
aseqdump -p 16

>>
Waiting for data. Press Ctrl+C to end.
Source  Event                  Ch  Data
 16:0   Note on                 0, note 50, velocity 37
 16:0   Note off                0, note 50
 16:0   Note on                 0, note 45, velocity 24
 16:0   Note off                0, note 45

```

In Reaper:
 Go to Options > Preferences > Audio > MIDI Devices.
    If the MIDI input device (hw:U49) is disabled:
        Right-click > Enable Input.
        Right-click > Enable for Control.

You should now see the midi lines activated on keypress. 
Alt+b in virtual midi keyboard should show them as well.

Now install the Virtual instrument (VSTi) e.g. Vital: https://aur.archlinux.org/packages/vital-synth
Restart reaper and it should be available in the FX tab of the track.
