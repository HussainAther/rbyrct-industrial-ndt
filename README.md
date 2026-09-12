# rbyrct-industrial-ndt

Synthetic proof-of-concept for adaptive sparse-view tomography in industrial non-destructive testing.

## Question
Can adaptive acquisition recover pore/crack defects in a manufactured part using fewer projection views?

## Phantom
Machined annular part + three pores + one crack-like low-density defect.

## Metrics
MSE, SSIM, defect IoU, defect recall, angle/ray count, runtime.

## Run
```bash
python experiments/experiment_001.py
```
