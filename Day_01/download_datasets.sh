# Create data directory
mkdir -p datasets

# naver sentiment corpus
wget -O ./datasets/naver_train.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
wget -O ./datasets/naver_test.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt

# namuwiki
wget -O ./datasets/namuwiki170327.7z https://dataserver.xyz/files/namuwiki170327.7z
7z x namuwiki170327.7z
