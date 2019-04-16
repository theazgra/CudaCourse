#pragma once

#include <SOIL.h>
#include <stdio.h>

enum ImageType
{
    ImageType_GrayScale_8bpp,
    ImageType_GrayScale_16bpp,
    ImageType_RGB_24bpp
};

inline int channels_per_image_type(const ImageType &it)
{
    switch (it)
    {
    case ImageType_GrayScale_8bpp:
    case ImageType_GrayScale_16bpp:
        return 1;
    case ImageType_RGB_24bpp:
        return 3;
    default:
        return -1;
    }
}

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
    ImageType image_type() const;

    unsigned char *data() const;
    unsigned char *at(const int x, const int y) const;
    void print_pixel(const int x, const int y) const;
};