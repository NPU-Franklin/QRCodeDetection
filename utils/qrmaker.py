import logging
import os

import qrcode
import zxing

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

data = "Hello world!"
save_path = "../data"

if __name__ == '__main__':
    logging.info("Begin generating...")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    code = qrcode.QRCode(
        version=4,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=100,
        border=5
    )
    code.add_data(data)
    code.make(fit=True)

    image = code.make_image()
    image.save(os.path.join(save_path, "base.jpg"))

    logging.info("Generating complete!")

    logging.info("Testing...")
    reader = zxing.BarCodeReader()
    data_decoded = reader.decode(os.path.join(save_path, "base.jpg")).parsed
    logging.info(f"Data containing: {data_decoded}")
