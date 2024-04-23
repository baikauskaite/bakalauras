BASE_DIR="/home/viktorija/bakalaurinis/sent-bias"
DIRECTORY="${BASE_DIR}/tests/english"

for file in "$DIRECTORY"/*-fr-names.jsonl
do
  echo $file
  filename=$(basename "$file")

  if [ -f "$file" ] && [[ $filename == weat* ]]; then
    python $BASE_DIR/scripts/generate_basic_contexts.py $file

    if [ $? -ne 0 ]; then
      echo "generate_basic_contexts.py exited with an error. Stopping the script."
      exit 1
    fi

  fi
done