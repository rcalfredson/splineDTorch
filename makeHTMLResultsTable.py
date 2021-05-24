import argparse
import calendar
from glob import glob
import json
import os

from util import meanConfInt

p = argparse.ArgumentParser(
    description="Create HTML table of results from" " a learning experiment."
)
p.add_argument(
    "error_dir", help="Path to folder containing error files in JSON format."
)
opts = p.parse_args()
error_files = glob(os.path.join(opts.error_dir, "*_errors.json"))
headers = (
    "name",
    "mean abs. error",
    "max abs. error",
    "mean rel. error",
    "mean rel. error, 0-10 eggs",
    "mean rel. error, 11-40 eggs",
    "mean rel. error, 41+ eggs",
)
json_keys = (
    "mean_abs_error",
    "max_abs_error",
    "mean_rel_error",
    "mean_rel_error_0to10",
    "mean_rel_error_11to40",
    "mean_rel_error_41plus",
)
months = [m.lower() for m in calendar.month_abbr[1:]]


def generate_name_str(filename):
    num_epochs = filename.split("epochs")[0].split("_")[-1]
    date, time = filename.split("_")[-2].split(" ")
    split_date = date.split("-")
    date_num = split_date[-1]
    month = months[int(split_date[1]) - 1]
    year = split_date[0]
    date_str = f"{date_num}-{month}-{year}"
    host = filename.split("_")[-3].split("Yang-Lab-")[-1]
    return f"{num_epochs} epochs, {date_str}, {'-'.join(time.split('-')[:2])}, {host}"

errors_by_type = {k: [] for k in json_keys}

with open("error_table.html", "w") as f:
    f.write('<table style="width: 100%;" border="0">\n')
    f.write("<tbody><tr>")
    for header in headers:
        f.write(f"<td><strong>{header}</strong></td>\n")
    f.write("<tr>\n")
    for fpath in error_files:
        f.write("<tr>")
        f.write(f"<td>{generate_name_str(fpath)}</td>\n")
        with open(fpath) as f_errs:
            errs = json.load(f_errs)
            for k in json_keys:
                f.write(f"<td>{errs[k]:.3f}</td>\n")
                errors_by_type[k].append(errs[k])
        f.write("</tr>")
    f.write("</tbody>\n")
    f.write("</table>")

print('Means of errors by type:')
for k in errors_by_type:
    print(f'{k}:')
    print(meanConfInt(errors_by_type[k], asDelta=True))
