#!/usr/bin/env sh
python io_data.py --label mspec
python io_data.py --dfeat_winlen 3 mspec
python io_data.py --dfeat_winlen 3 lmfcc
python io_data.py lmfcc
