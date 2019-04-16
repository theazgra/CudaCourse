#include <image.h>

Image::Image(const char *filename, ImageType type)
{
    this->type = type;

    _data = SOIL_load_image(filename, &_width, &_height, &_channels, channels_per_image_type(type));
}

Image::~Image()
{
    SOIL_free_image_data(_data);
}

int Image::width() const
{
    return _width;
}
int Image::height() const
{
    return _height;
}
int Image::channel_count() const
{
    return _channels;
}

ImageType Image::image_type() const
{
    return type;
}

unsigned char *Image::data() const
{
    return _data;
}

unsigned char *Image::at(int x, int y) const
{
    return (_data + (((y * _width) + x) * _channels));
}

void Image::print_pixel(const int x, const int y) const
{
    switch (type)
    {
    case ImageType_GrayScale_8bpp:
        printf("%3u", *at(x, y));
        break;
    case ImageType_RGB_24bpp:
    {
        unsigned char *px = at(x, y);
        printf("%3u%3u%3u", px[0], px[1], px[2]);
    }
    break;
    }
}
