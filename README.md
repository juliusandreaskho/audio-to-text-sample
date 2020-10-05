# audio-to-text-sample

This project contains 3 folders and 4 main code, as follows.

1. models/svc which contains the model used in this repo project.

2. test_sound_mono contains the test set data used in prediction (audio_to_text.py).

3. train_sound_mono contains the 50 folders of training set audio data, 
which folder contains 8 files which represent the speech output of that words.

Main Code:

4. machine_train_wavestruct.py --> training process.

5. audio_to_text.py --> prediction by model.

6. svc_machine_train.py --> documentation of the experiments which still have many errors.

7. audio_to_text_google.py --> experiments I made using Google TTS service.

To execute the program, please ONLY run the "machine_train_wavestruct.py" and then "audio_to_text.py".
Feel free to learn from, give feedback to this repo, and Enjoy the Program :)

Note:
There are 11 unused folders in this repo and will be deleted as soon as possible. 
The folders are named "awan, baju, bangun, bapak, berjalan, berlari, bintang, bulan, cangkir, celana, datang".
