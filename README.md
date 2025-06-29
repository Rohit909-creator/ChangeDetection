# 🔁 ChangeDetection 2.0 — SAM-Powered Difference Detector

``A simple math based spot the difference model, enhanced using SAM model``

Okay, so basically this is my new attempt at making a smarter change detection model — not just the boring "something changed" kind, but more like:

> “Hey, this button shifted to the left... and also, this part here got replaced with something new.”

The whole idea is to **segment out what actually changed** between two images (UI, scenes, whatever) using a clean combo of:
- basic **image difference** difference = img1 - img2
- **contour-based region extraction** on the difference mask, to point prompt the SAM model 
- and **SAM (Segment Anything Model)** to get the actual object segments


---

### 💡 What's Done So Far

I’ve already got the core pipeline running:
- Take two images (before and after)
- Do a diff to get changed regions
- Use **contour detection** to extract areas that seem to have changed
- Pass those as input points into **SAM** to get exact masks of what changed

So visually, the pipeline looks like this:

```plaintext
Image A   Image B
   │         │
   └──▶ Diff Mask ──▶ Contour Detection ──▶ SAM Segments
                                   │
                         (Region-level Change Maps)
```
**Btw there is no changedetection1.0 algorithm on my github, cause it was my first internship project, and I gotta respect NDA**