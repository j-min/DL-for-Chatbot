if [[ `uname` == 'Darwin' ]]; then
    curl -fLo /tmp/ngrok.zip https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-darwin-amd64.zip
    unzip -j /tmp/ngrok.zip
elif [[ `uname` == 'Linux' ]]; then
    curl -fLo /tmp/ngrok.zip https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
    unzip -j /tmp/ngrok.zip
else
    # Unknown OS
    echo "다음 페이지에서 직접 다운로드 해주세요 => https://ngrok.com/"
fi
