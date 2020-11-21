# defog

 

Ushape_ResNet( 

  (encode1): Sequential( 

    (0): ResDilationBlock() 

    (1): ResDilationBlock() 

    (2): ResDilationBlock() 

  ) 

  (encode2): Sequential( 

    (0): ResDilationBlock() 

    (1): ResDilationBlock() 

    (2): ResDilationBlock() 

  ) 

  (encode3): Sequential( 

    (0): ResDilationBlock() 

    (1): ResDilationBlock() 

    (2): ResDilationBlock() 

  ) 

  (encode4): Sequential( 

    (0): ResBlock() 

    (1): ResBlock() 

    (2): ResBlock() 

  ) 

  (deconv1): ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2)) 

  (decode1): Sequential( 

    (0): ResBlock() 

    (1): ResBlock() 

    (2): ResBlock() 

  ) 

  (deconv2): ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2)) 

  (decode2): Sequential( 

    (0): ResBlock() 

      (shortcut): Sequential( 

        (0): Conv2d(48, 32, kernel_size=(1, 1), stride=(1, 1), bias=False) 

        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) 

    (1): ResBlock() 

    (2): ResBlock() 

  ) 

  (deconv3): ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2)) 

  (decode3): Sequential( 

    (0): ResBlock() 

    (1): ResBlock() 

    (2): ResBlock() 

  ) 

  (pre): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 

  (sigmoid): Sigmoid() 

) 

 

ResBlock( 

      (left): Sequential( 

        (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 

        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) 

        (2): ReLU(inplace=True) 

        (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 

        (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) 

      ) 

 

ResDilationBlock( 

      (left_1): Sequential( 

        (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 

        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) 

        (2): ReLU(inplace=True) 

        (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 

        (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) 

      ) 

      (left_2): Sequential( 

        (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 

        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) 

        (2): ReLU(inplace=True) 

        (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False) 

        (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) 

        (5): ReLU(inplace=True) 

        (6): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(5, 5), dilation=(5, 5), bias=False) 

        (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) 

      ) 

      (decay): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False) 

      (shortcut): Sequential() 

    ) 

  ) 

 

 

 
