DIRECTORY="/home/viktorija/bakalaurinis/sent-bias-master/tests/english"

for file in "$DIRECTORY"/*
do
  echo $file
  filename=$(basename "$file")

  if [ -f "$file" ] && [[ $filename == sent* ]]; then
    python translate.py --test_name $file

    if [ $? -ne 0 ]; then
      echo "translate.py exited with an error. Stopping the script."
      exit 1
    fi

  fi
done