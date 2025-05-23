



### Task III.A: Simulated Image Super-Resolution

#### Models Implemented
1. **SRCNN** (Small and Large variants)
   - Basic CNN architecture for super-resolution
2. **ESPCN**
   - Efficient Sub-Pixel Convolutional Network
3. **SRGAN**
   - Generative adversarial network approach
4. **SRResNet**
   - Residual network based approach

#### Evaluation Metrics
Compared models using:
- Mean Squared Error (MSE)
- Structural Similarity Index (SSIM)
- Peak Signal-to-Noise Ratio (PSNR)

#### Results

| Model         | PSNR (dB)         | SSIM              | MSE                   |
|---------------|-------------------|-------------------|-----------------------|
| SRCNN-small   | 42.01907361871431 | 0.974398921251297 | 6.332411169569241e-05 |
| SRCNN-large   | 42.10237243971328 | 0.974830070137977 | 6.21188970944786e-05  |
| ESPCN         | **42.28710515495454** | **0.975952082574367** | **5.95304630096507e-05**  |
| SRGAN         | 40.48610935549050 | 0.958761047363281 | 9.002151059030438e-05 |
| SRResNet      | 41.65625927735547 | 0.971796224296093 | 6.882382733238046e-05 |

### Task III.B: Real Telescope Image Super-Resolution

#### Approach
- Fine-tuned models from Task III.A on real HSC/HST data
- Employed transfer learning and data augmentation techniques

#### Evaluation Metrics
Compared models using:
- Mean Squared Error (MSE)
- Structural Similarity Index (SSIM)
- Peak Signal-to-Noise Ratio (PSNR)

#### Results

| Model         | PSNR (dB)         | SSIM               | MSE                   |
|---------------|-------------------|--------------------|-----------------------|
| SRCNN-small   | 34.09434717290395 | 0.8582045892874400 | 0.0008473546695313416 |
| SRCNN-large   | 35.21323692730531 | 0.8576015591621399 | **0.0007645150971560118** |
| ESPCN         | **35.39417348870955** | **0.8734989364941915** | 0.0007788492572823694 |
| SRResNet      | 35.24461601752956 | 0.8417196949323018 | 0.0007827904955775011 |

#### Future Work
- Look into diffusion model based approaches
- Look into training larger model
- Implement a special augmentation technique called CutBlur, which is designed for Image Super-Resolution






