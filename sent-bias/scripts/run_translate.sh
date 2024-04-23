BASE_DIR="/home/viktorija/bakalaurinis/sent-bias"
DIRECTORY="${BASE_DIR}/tests/english"

for file in "$DIRECTORY"/*-fr-names.jsonl
do
  echo $file
  filename=$(basename "$file")

  if [ -f "$file" ] && [[ $filename == sent* ]]; then
    python $BASE_DIR/scripts/translate.py --test_name $filename

    if [ $? -ne 0 ]; then
      echo "translate.py exited with an error. Stopping the script."
      exit 1
    fi

  fi
done