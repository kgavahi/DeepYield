"""
Created on Sun Jan 26 17:00:27 2020

@author: kayhangavahi@gmail.com
"""

import datetime
import time
import urllib
import os
import requests
from pyhdf.SD import SD, SDC
#from netCDF4 import Dataset

def LetMeSleep(sec):
    print("Connection refused by the server..")
    print("It can be due to sending multiple requests")
    print(f"Let me sleep for {sec} seconds")
    print("ZZzzzz...")
    time.sleep(sec)
    print("Was a nice sleep, now let me continue...")    
def readCredentials(file):
    with open(file) as f:
        lines = f.readlines()
    return lines
def downloadFile(f, save_name):
    while True:
        try:
            response = requests.get(f.strip(), stream=True)
            if response.status_code != 200:
                print("Verify that your username and password are correct")
            else:
                with open(save_name, 'wb') as d:
                    d.write(response.content)
                print(f'Downloaded file: {save_name}')
                break
        except:
            LetMeSleep(5)
    
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))
def read_webpage(filex):
    while True:
        try:
            with urllib.request.urlopen(filex) as f:
                r = f.read().decode('utf-8')
            break
        except:
            LetMeSleep(5)
    return r
def DownloadList_MODIS(username, password, date_start, date_end, earthData_name, earthData_version):

    save_dir = f'{earthData_name}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    path_netrc = os.path.expanduser("~/.netrc")
    if os.path.exists(path_netrc):
        os.remove(path_netrc)
        
    with open(path_netrc, 'w') as f:
        f.write(f"machine urs.earthdata.nasa.gov\nlogin {username}\npassword {password}")    

    # Convert dates to datetime objects
    date_start = datetime.datetime.strptime(date_start, '%Y-%m-%d').date()
    date_end = datetime.datetime.strptime(date_end, '%Y-%m-%d').date()

    
    
    sattelite_mapping = {
        'MCD': 'MOTA',
        'MYD': 'MOLA',
        'MOD': 'MOLT'
    }

    sattelite = sattelite_mapping.get(earthData_name[:3], 'unknown')
        
    filex = f"https://e4ftl01.cr.usgs.gov/{sattelite}/{earthData_name}.{earthData_version}/"
    page = read_webpage(filex)
    
       
    lines = page.split('\n')
    matches = [s for s in lines if "folder.gif" in s]
    date_strings = [s[s.find('href="') + 6: s.find('href="') + 6 + 4 + 2 + 2 + 2] for s in matches]
    dateList = [datetime.datetime.strptime(s, '%Y.%m.%d').date() for s in date_strings]
    


    # Only select available dates for the dataset
    first_date = nearest(dateList, date_start)
    last_date  = nearest(dateList, date_end)
      
    dateList_to_download = [date for date in dateList if first_date <= date <= last_date]

    
    # 14 tiles that cover the CONUS
    conus_tiles = ["h08v04","h08v05","h08v06",
                    "h09v04","h09v05","h09v06",
                     "h10v04","h10v05","h10v06",
                     "h11v04","h11v05","h12v04",
                     "h12v05","h13v04"]    



    for i, date in enumerate(dateList_to_download):           
        date_str = date.strftime("%Y.%m.%d")
        filex = f"https://e4ftl01.cr.usgs.gov/{sattelite}/{earthData_name}.{earthData_version}/{date_str}/"
        page = read_webpage(filex)
        page_lines = page.split('\n')
        hdf_files  = [line for line in page_lines if "hdf" in line]
        
        conus_files = [s for s in hdf_files if \
                any(substring in s for substring in conus_tiles)]

        if not conus_files:
            print(f'NO DATA AVAILABLE for {date}')
            continue

        start_ind = conus_files[0].find(earthData_name)
        end_ind   = conus_files[0].find('.hdf') + 4
        mylist = [filex + i[start_ind:end_ind] for i in conus_files]
        URLs = list(set(mylist))
           
        download(URLs, save_dir)
        print('    ',str((i+1)/len(dateList_to_download)*100)[:5] + ' % Completed')


def download(file_list, save_dir):
    
    

    file_list = sorted(file_list)

    
    # -----------------------------------------DOWNLOAD FILE(S)-------------------------------------- #
    # Loop through and download 14 files to the directory specified above, and keeping same filenames
    for i, f in enumerate(file_list):

        date_of_file = f.split('/')[5].replace('.','-')
        file_name = f.split('/')[-1].strip()
        path = os.path.join(save_dir, date_of_file)
        path = f'{save_dir}/{date_of_file}'

        if not os.path.exists(path):
            os.mkdir(path)
        save_name = f'{path}/{file_name}'

        if os.path.exists(save_name):
            try:

                f = SD( save_name , SDC.READ)
                f.end()
                continue
            
            except:
                print('Damaged file encountered, redownloading...')

        # Create and submit request and download file
        downloadFile(f, save_name)

        
        
def main():
    
    
    cred = readCredentials('credentials.txt')
    username   = cred[0]
    password   = cred[1]
    start_date = '1950-01-01'
    end_date   = '2018-2-2'
    product    = 'MOD13A1'
    version    = '006'

    
    
       
    DownloadList_MODIS(username, 
                       password, 
                       start_date, 
                       end_date, 
                       product, 
                       version)
    

if __name__ == '__main__':

    main()