#pragma once

#include <SOIL.h>
#include <stdio.h>

enum ImageType
{
    ImageType_GrayScale_8bpp = 1,
    ImageType_RGB_24bpp = 3
};

class Image
{
  private:
    unsigned char *_data;
    int _width;
    int _height;
    int _channels;
    ImageType type;

  public:
    Image(const char *filename, ImageType type);
    ~Image();

    int width() const;
    int height() const;
    int channel_count() const;

    unsigned char *data() const;
    unsigned char *at(const int x, const int y) const;
    void print_pixel(const int x, const int y) const;
};