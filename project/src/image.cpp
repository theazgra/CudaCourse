#include <image.h>

Image::Image(const char *filename, ImageType type)
{
    this->type = type;

    FreeImage_Initialise();
    FREE_IMAGE_FORMAT fileFormat = FreeImage_GetFileType(filename);
    _data = FreeImage_Load(fileFormat, filename);

    _channels = channels_per_image_type(type);
    _width = FreeImage_GetWidth(_data);
    _height = FreeImage_GetHeight(_data);
    _pitch = FreeImage_GetPitch(_data);
}

Image::~Image()
{
    FreeImage_Unload(_data);
    FreeImage_DeInitialise();
}

int Image::width() const
{
    return _width;
}

int Image::height() const
{
    return _height;
}

size_t Image::pitch() const
{
    return _pitch;
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
    BYTE *dataPtr = FreeImage_GetBits(_data);
    return dataPtr;
}