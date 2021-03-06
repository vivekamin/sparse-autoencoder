# sparse-autoencoder
"Autoencoding" is a data compression algorithm where the compression and decompression functions are 1) data-specific, 2) lossy, and 3) learned automatically from examples rather than engineered by a human. Additionally, in almost all contexts where the term "autoencoder" is used, the compression and decompression functions are implemented with neural networks.[learn more here](https://blog.keras.io/building-autoencoders-in-keras.html)


 Trained Sparse Autoencoder for sparse parameter p = [0.01, 0.1, 0.5, 0.8]
 ## Fetures(Weights) learned by encoder
![weights for p 0 1](https://user-images.githubusercontent.com/25477734/38649026-374bd32e-3da9-11e8-933f-79a816448bef.png)
![weights for p 0 01](https://user-images.githubusercontent.com/25477734/38649027-3760d210-3da9-11e8-9d7b-0028c72d6389.png)
![weights for p 0 5](https://user-images.githubusercontent.com/25477734/38649028-37773d52-3da9-11e8-990b-6d45ee8ec277.png)
![weights for p 0 8](https://user-images.githubusercontent.com/25477734/38649029-378afb26-3da9-11e8-9a71-e19686d0c8af.png)

## Best image reconstruction happened for P = 0.1
![input for p 0 1](https://user-images.githubusercontent.com/25477734/38649079-7c64f56c-3da9-11e8-8909-7c9a148b9154.png)
![output for p 0 1](https://user-images.githubusercontent.com/25477734/38649080-7c784e28-3da9-11e8-8e23-8d84154bb46e.png)
