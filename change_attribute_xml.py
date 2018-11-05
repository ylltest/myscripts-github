import os
import xml.etree.cElementTree as et

input_file_dir="C:\\Users\\Administrator.X36KKQ2UTQSEZ5O\\Desktop\\xmltest"    # 待读取文件存放路径
xml_dir="C:\\Users\\Administrator.X36KKQ2UTQSEZ5O\\Desktop\\xml-ok\\"  # 输出xml文件保存路径

def alter_file(file):
    tree=et.parse(file)
    root=tree.getroot()
    for node in root.iter('name'):
        new_name = 'person'
        node.text = new_name
        # node.set("updated", "yes")
    print(xmi_name)
    tree.write(xml_dir + xmi_name + '.xml')


for f in os.listdir(input_file_dir):
    xmi_name = str(f[:-4])
    if alter_file(input_file_dir+"\\"+f) == False:
        break