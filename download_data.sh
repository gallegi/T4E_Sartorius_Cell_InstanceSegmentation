wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1M7C6FHgBnCjxo7qTmz3Zn0tHTwznEEGy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1M7C6FHgBnCjxo7qTmz3Zn0tHTwznEEGy" -O data/annotation_semisupervised_round1.zip && rm -rf /tmp/cookies.txt
unzip data/annotation_semisupervised_round1.zip -d data/

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1coW2dINEvQba50vAaOSvURCQa_8L4CHR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1coW2dINEvQba50vAaOSvURCQa_8L4CHR" -O data/annotation_semisupervised_round2.zip && rm -rf /tmp/cookies.txt
unzip data/annotation_semisupervised_round2.zip -d data/

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=133nHthlCY-qnnXG35GQU8iyoxgoPM7bP' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=133nHthlCY-qnnXG35GQU8iyoxgoPM7bP" -O data/images.zip && rm -rf /tmp/cookies.txt
unzip data/images.zip -d data/
