In this folder, the raw images need to be stored. 

You can download and decompressed them, using nex commands:

```bash
cd src/data/sets/raw/images
```
```
sudo wget http://lstm.seas.harvard.edu/latex/data/formula_images_processed.tar.gz
```
```
sudo tar -zxvf formula_images_processed.tar.gz
```

Then, you have to copy and paste the images at the level of this file.
Suggested command to copy many files:
```
find /src/data/sets/raw/images/formula_images_processed/ -name "*.png" -exec cp -uf "{}" /src/data/sets/raw/images/ \;
```
