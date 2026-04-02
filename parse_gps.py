import os
import glob
import json
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    if not exif_data:
        return None
    for key, value in exif_data.items():
        name = TAGS.get(key, key)
        if name == 'GPSInfo':
            gps_info = {}
            for t in value:
                sub_decoded = GPSTAGS.get(t, t)
                gps_info[sub_decoded] = value[t]
            return gps_info
    return None

def get_decimal_from_dms(dms, ref):
    if dms is None or ref is None:
        return 0.0
    # Handle PIL rational types if necessary
    try:
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
    except TypeError:
        # If they are already float
        degrees, minutes, seconds = dms
    dec = degrees + minutes/60.0 + seconds/3600.0
    if ref in ['S', 'W']:
        dec = -dec
    return dec

def main():
    image_dir = r'c:\cg\Project\python\slam3r\SLAM3R\trans-gps\images'
    all_files = glob.glob(os.path.join(image_dir, '*.JPG')) + glob.glob(os.path.join(image_dir, '*.jpg'))
    files = sorted(list(set(all_files)))
    points = []
    
    for f in files:
        try:
            gps_info = get_exif_data(f)
            if gps_info:
                lat = get_decimal_from_dms(gps_info.get('GPSLatitude'), gps_info.get('GPSLatitudeRef'))
                lon = get_decimal_from_dms(gps_info.get('GPSLongitude'), gps_info.get('GPSLongitudeRef'))
                
                # Check for altitude
                alt = 100.0
                if 'GPSAltitude' in gps_info:
                    alt_val = gps_info['GPSAltitude']
                    try:
                        alt = float(alt_val)
                    except:
                        alt = 100.0
                
                points.append({
                    'file': os.path.basename(f),
                    'lon': lon,
                    'lat': lat,
                    'alt': alt
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    out_path = r'c:\cg\Project\python\slam3r\SLAM3R\tmp\gps_path.json'
    with open(out_path, 'w') as out:
        json.dump(points, out, indent=2)
    print(f"Saved {len(points)} GPS points to {out_path}.")

if __name__ == '__main__':
    main()
