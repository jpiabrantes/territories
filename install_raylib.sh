wget https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_linux_amd64.tar.gz
tar -xzf raylib-5.5_linux_amd64.tar.gz
cd raylib-5.5_linux_amd64

# Copy headers and library to system paths (optional)
sudo cp -r include/* /usr/local/include/
sudo cp lib/libraylib.a /usr/local/lib/
sudo ldconfig

# clean up
rm raylib-5.5_linux_amd64.tar.gz
rm -rf raylib-5.5_linux_amd64
