DIRECTORY="/home/viktorija/bakalaurinis/sent-bias-master/tests/english"

for file in "$DIRECTORY"/*
do
  echo $file
  filename=$(basename "$file")

  if [ -f "$file" ] && [[ $filename == weat* ]]; then
    python generate_basic_contexts.py $file

    if [ $? -ne 0 ]; then
      echo "generate_basic_contexts.py exited with an error. Stopping the script."
      exit 1
    fi

  fi
done