BASE_DIR="/home/viktorija/bakalaurinis/sent-bias"
DIRECTORY="${BASE_DIR}/../context-debias/data/french"

for file in "$DIRECTORY"/*
do
  if [[ "$file" =~ \.fr$ ]] || [[ "$file" =~ \stereotypes.txt$ ]]; then
    continue
  fi
  echo $file
  filename=$(basename "$file")

  if [ -f "$file" ]; then
    python $BASE_DIR/scripts/translate.py --test_name $filename --dir "${BASE_DIR}/../context-debias/data"

    if [ $? -ne 0 ]; then
      echo "translate.py exited with an error. Stopping the script."
      exit 1
    fi

  fi
done