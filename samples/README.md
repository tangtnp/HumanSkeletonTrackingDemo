# Sample videos

Put a short video clip of one or more people here (e.g. `walk.mp4`) and run:

```bash
python src/main.py --source samples/walk.mp4 --save output.mp4
```

Good demo clips: someone walking across the frame (shows tracking + speed +
trajectory) or doing squats from the side (shows joint angle + rep counter).

No video on hand? Use your webcam instead:

```bash
python src/main.py --source 0
```
