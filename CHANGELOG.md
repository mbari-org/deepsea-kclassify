# DeepSea Keras Image Classifier Changelog

## [1.0.5](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.4...v1.0.5) (2020-11-03)


### Bug Fixes

* bGR to RGB color conversion, added PIL library and tf2 iterator ([2c13fbe](http://bitbucket.org/mbari/deepsea-kclassify/commits/2c13fbecb1904ea563466b56e30a56ea0657d718))
* correct tf2 tensor metric reference ([5908809](http://bitbucket.org/mbari/deepsea-kclassify/commits/590880907f98af9bb8286b73c72d57df3631f8ed))
* fixed merge conflict ([ec77542](http://bitbucket.org/mbari/deepsea-kclassify/commits/ec775426e3f3d64ad2568303b003630f8849ecc3))

## [1.0.4](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.3...v1.0.4) (2020-11-03)


### Performance Improvements

* migrated to latest tensorflor 2.3.1 ([71ff09f](http://bitbucket.org/mbari/deepsea-kclassify/commits/71ff09f1e7b159fe041aa00f90732e52bdcc710a))

## [1.0.3](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.2...v1.0.3) (2020-10-13)


### Bug Fixes

* replace WANDB_RUN with WANDB_RUN_GROUP per latest wandb api ([8df4b61](http://bitbucket.org/mbari/deepsea-kclassify/commits/8df4b61fcbf5c33381441ed6aa33e369ab52b7ad))

## [1.0.2](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.1...v1.0.2) (2020-10-03)


### Bug Fixes

* correct normalize for VGG models ([3e6e929](http://bitbucket.org/mbari/deepsea-kclassify/commits/3e6e929c4f66bb49bc0975d1c9eddbce634238b0))
* correct normalize logic ([cf6a169](http://bitbucket.org/mbari/deepsea-kclassify/commits/cf6a169de2469ff6767e152a15d12d237266619a))
* correct per tf version ([290d1fa](http://bitbucket.org/mbari/deepsea-kclassify/commits/290d1fa80044e187deb1e6f9241c4c1f333f1081))

## [1.0.1](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.0...v1.0.1) (2020-07-25)


### Bug Fixes

* **image_pyfunc.py:** install tf through pip as its not in normal conda channels ([90228a9](http://bitbucket.org/mbari/deepsea-kclassify/commits/90228a958b097d39ff26f4664eb7197e561bc615))
* **mlproject:** correct bucket name ([5d2750c](http://bitbucket.org/mbari/deepsea-kclassify/commits/5d2750c0fbd56223e4a7053fc22785fb7e95b2e4))

# 1.0.0 (2020-07-08)


### Bug Fixes

* **conf.py:** revert fine_tune_at for all models to -1 ([1c46c02](http://bitbucket.org/mbari/deepsea-kclassify/commits/1c46c028a38536e685f554a0b4e60fe238e8796d))
* **image_pyfunc.py:** add placed holder for empty mean/std ([9ccd531](http://bitbucket.org/mbari/deepsea-kclassify/commits/9ccd5317d0afd6388fb3c9c46ad7c9f89199b3cb))
* **mlproject:** removed unsupported balance arg ([7b21452](http://bitbucket.org/mbari/deepsea-kclassify/commits/7b21452678fba3174eae57796657c90d01b7db0e))
* **mlproject:** removed unused data balance option ([e8d078a](http://bitbucket.org/mbari/deepsea-kclassify/commits/e8d078aa0bf112c98e6334b0fb0d2765aa0f2353))
* **mlproject:** test_tar corrected to val_tar ([07b9be2](http://bitbucket.org/mbari/deepsea-kclassify/commits/07b9be2d6caba82b042bb9f9c23abb4f437851f6))
* **requirements.txt:** added missing dependency ([77400e5](http://bitbucket.org/mbari/deepsea-kclassify/commits/77400e5db3896a5c2c9b6cb73515447540f53c59))
* **requirements.txt:** bumped wandb to available version ([fb39af5](http://bitbucket.org/mbari/deepsea-kclassify/commits/fb39af508d744d8d3d43dcb5abf7dffa4fdbf5bc))
* **test:** working model train test ([7eddc80](http://bitbucket.org/mbari/deepsea-kclassify/commits/7eddc809d272421c9225cbfd78109a518f8bed7a))
* **train.py:** add in wanddb callback during training if wandb available ([098254c](http://bitbucket.org/mbari/deepsea-kclassify/commits/098254cc9beee735e3b36876eb7042c1d36ff05f))
* **train.py:** correct arg.normalize to json ([df10d85](http://bitbucket.org/mbari/deepsea-kclassify/commits/df10d8529c3484e07230a4fe9ba7aa6adea70b12))
* **train.py:** correct argument ([c622154](http://bitbucket.org/mbari/deepsea-kclassify/commits/c622154787965945417cc71fdee6a57f14735c8c))
* **train.py:** correct generator for mean/std ([ed674a7](http://bitbucket.org/mbari/deepsea-kclassify/commits/ed674a7415a5ec2a99f0963af5df91818143a7f1))
* **train.py:** fixed wandb init bug and Stopping append ([80a60f8](http://bitbucket.org/mbari/deepsea-kclassify/commits/80a60f8a93d0cef3a348ed206d91b714868572ef))
* **train.py:** function not member fix for normalize stats ([104e4da](http://bitbucket.org/mbari/deepsea-kclassify/commits/104e4da4ce64e6bef6b3aa07b58d3b2f934e1c28))
* **train.py:** normalize pred and eval fixes ([0b4494c](http://bitbucket.org/mbari/deepsea-kclassify/commits/0b4494c2218432bcd0bd89a97947bd74d63e25c5))
* **train.py:** remove shuffle to support cm/pr at end ([81102ed](http://bitbucket.org/mbari/deepsea-kclassify/commits/81102ed6bea4b313ea8a6f8d7296325b19bcba01))
* **transfer_model.py:** fix surgeon unfreeze bug ([feb16fb](http://bitbucket.org/mbari/deepsea-kclassify/commits/feb16fb5bc3e65304ea8737a4722d91e51b949c8))
* added missing file ([b9c3f1b](http://bitbucket.org/mbari/deepsea-kclassify/commits/b9c3f1b8625e8c15d21308c3f7c995a5975e8ea2))
* fixed bug in wand pr/roc/cm logging ([ba8befd](http://bitbucket.org/mbari/deepsea-kclassify/commits/ba8befd456df1f0002986ecfe8dc29c83eb7b499))
* handle missing val dir and remove var with outer scope name ([26cb658](http://bitbucket.org/mbari/deepsea-kclassify/commits/26cb658601d1c70f481eb71fb4fa93588b0bc20a))
* reduced early stop patience to avoid prolonged runs ([53bc567](http://bitbucket.org/mbari/deepsea-kclassify/commits/53bc567632759f85db6053a2db510ad5533f29d5))
* removed imblearn until sklearn issue is resolved ([85c4353](http://bitbucket.org/mbari/deepsea-kclassify/commits/85c435304d2bb88a6b4f292f8532a4de9329ed22))
* removed indexing ([4dd8d43](http://bitbucket.org/mbari/deepsea-kclassify/commits/4dd8d43eed605f7d95113194120222b180a55d23))
* save labels correctly from indexed dictionary ([e24fa4c](http://bitbucket.org/mbari/deepsea-kclassify/commits/e24fa4c9f4ff1f9f198876451c9a28ebd6f6e2bb))

## [1.0.9](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.8...v1.0.9) (2020-06-24)


### Bug Fixes

* removed imblearn until sklearn issue is resolved ([85c4353](http://bitbucket.org/mbari/deepsea-kclassify/commits/85c435304d2bb88a6b4f292f8532a4de9329ed22))
* **requirements.txt:** bumped wandb to available version ([fb39af5](http://bitbucket.org/mbari/deepsea-kclassify/commits/fb39af508d744d8d3d43dcb5abf7dffa4fdbf5bc))

## [1.0.9](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.8...v1.0.9) (2020-06-24)


### Bug Fixes

* **requirements.txt:** bumped wandb to available version ([fb39af5](http://bitbucket.org/mbari/deepsea-kclassify/commits/fb39af508d744d8d3d43dcb5abf7dffa4fdbf5bc))

## [1.0.8](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.7...v1.0.8) (2020-06-06)


### Bug Fixes

* fixed bug in wand pr/roc/cm logging ([ba8befd](http://bitbucket.org/mbari/deepsea-kclassify/commits/ba8befd456df1f0002986ecfe8dc29c83eb7b499))

## [1.0.7](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.6...v1.0.7) (2020-06-05)


### Bug Fixes

* removed indexing ([4dd8d43](http://bitbucket.org/mbari/deepsea-kclassify/commits/4dd8d43eed605f7d95113194120222b180a55d23))

## [1.0.6](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.5...v1.0.6) (2020-05-13)


### Bug Fixes

* save labels correctly from indexed dictionary ([e24fa4c](http://bitbucket.org/mbari/deepsea-kclassify/commits/e24fa4c9f4ff1f9f198876451c9a28ebd6f6e2bb))

## [1.0.5](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.4...v1.0.5) (2020-05-08)


### Bug Fixes

* handle missing val dir and remove var with outer scope name ([26cb658](http://bitbucket.org/mbari/deepsea-kclassify/commits/26cb658601d1c70f481eb71fb4fa93588b0bc20a))

## [1.0.4](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.3...v1.0.4) (2020-03-17)


### Bug Fixes

* reduced early stop patience to avoid prolonged runs ([53bc567](http://bitbucket.org/mbari/deepsea-kclassify/commits/53bc567632759f85db6053a2db510ad5533f29d5))

## [1.0.3](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.2...v1.0.3) (2020-03-14)


### Bug Fixes

* **train.py:** fixed wandb init bug and Stopping append ([80a60f8](http://bitbucket.org/mbari/deepsea-kclassify/commits/80a60f8a93d0cef3a348ed206d91b714868572ef))

## [1.0.2](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.1...v1.0.2) (2020-03-13)


### Bug Fixes

* **mlproject:** test_tar corrected to val_tar ([07b9be2](http://bitbucket.org/mbari/deepsea-kclassify/commits/07b9be2d6caba82b042bb9f9c23abb4f437851f6))

## [1.0.1](http://bitbucket.org/mbari/deepsea-kclassify/compare/v1.0.0...v1.0.1) (2020-03-12)


### Bug Fixes

* added missing file ([b9c3f1b](http://bitbucket.org/mbari/deepsea-kclassify/commits/b9c3f1b8625e8c15d21308c3f7c995a5975e8ea2))

# 1.0.0 (2020-03-12)


### Bug Fixes

* added missing file ([b9c3f1b](http://bitbucket.org/mbari/deepsea-kclassify/commits/b9c3f1b8625e8c15d21308c3f7c995a5975e8ea2))
* **test:** working model train test ([7eddc80](http://bitbucket.org/mbari/deepsea-kclassify/commits/7eddc809d272421c9225cbfd78109a518f8bed7a))
* **train.py:** add in wanddb callback during training if wandb available ([098254c](http://bitbucket.org/mbari/deepsea-kclassify/commits/098254cc9beee735e3b36876eb7042c1d36ff05f))

# 1.0.0 (2020-03-12)


### Bug Fixes

* **test:** working model train test ([7eddc80](http://bitbucket.org/mbari/deepsea-kclassify/commits/7eddc809d272421c9225cbfd78109a518f8bed7a))
* **train.py:** add in wanddb callback during training if wandb available ([098254c](http://bitbucket.org/mbari/deepsea-kclassify/commits/098254cc9beee735e3b36876eb7042c1d36ff05f))
