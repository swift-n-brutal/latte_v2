python test_wgan.py `
  --netg dcgan_g_d4_c128_o64_std0.02.prototxt `
  --netd dcgan_d_d4_c128_o64_std0.02_leaky0.2_nobnfirst.prototxt `
  --imfd e:\images\img_align_celeba `
  --imnm e:\images\img_align_celeba\names.txt `
  --dpit 10 `
  --tsit 20 `
  --snap 100 `
  --mxit 20000 `
  --svto results_leaky_small_nobnfirst `
  --qusz 4 `
  --dvid 1