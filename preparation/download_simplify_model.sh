#!/bin/bash

function download_model {
   FILENAME="$1"
   FILEURL="$2"
   FILEMD5="$3"
   echo "Downloading the Weight of simplifyNet..."
   wget -q --show-progress --continue -O "$FILENAME" -- "$FILEURL"

   echo -n "Checking integrity (md5sum)..."
   OS=`uname -s`
   if [ "$OS" = "Darwin" ]; then
      CHECKSUM=`cat $FILENAME | md5`
   else
      CHECKSUM=`md5sum $FILENAME | awk '{ print $1 }'`
   fi

   if [ "$CHECKSUM" != "$FILEMD5" ]; then
      echo "failed"
      echo "Integrity check failed. File is corrupt!"
      echo "Try running this script again and if it fails remove '$FILENAME' before trying again."
      exit 1
   fi 
   echo "ok"
}

download_model "Data/simplify_weight.pth" "https://doc-14-20-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/apu3eng8a546nf3mng6rc1na4m268ujj/1570788000000/11424217391890521276/*/1XdrLcTfmt2pCBkuqzlQRMOQ0WW3LaKZ2?e=download" "838e61faf600d3e585c686544be6b15e"
echo "Downloads finished successfully!"

