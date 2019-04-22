#pragma once

#include <FreeImage.h>
#include <stdio.h>

enum ImageType
{
    ImageType_GrayScale_8bpp,
    ImageType_GrayScale_16bpp,
    ImageType_RGB_24bpp
};

inline size_t pixel_byte_size(const ImageType &it)
{
    switch (it)
    {
    case ImageType_GrayScale_8bpp:
        return 1;
    case ImageType_GrayScale_16bpp:
        return 2;
    case ImageType_RGB_24bpp:
        return 3;
    default:
        return 0;
    }
}

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
    FIBITMAP *_data;

    int _width;
    int _height;
    int _channels;
    size_t _pitch;
    ImageType type;

public:
    Image(const char *filename, ImageType type);
    ~Image();

    size_t pitch() const;
    int width() const;
    int height() const;
    int channel_count() const;
    ImageType image_type() const;

    unsigned char *data() const;
};