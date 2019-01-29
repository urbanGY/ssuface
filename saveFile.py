#단순한 파일 입출력 예

f = open('data/image_list.txt',mode='wt',encoding='utf-8')
l = open('data/label_list.txt',mode='wt',encoding='utf-8')
for i in range(1848):
    name = 'data/pencilCase/test'
    name = name + str(i) + '.jpg'
    f.write(name+'\n')
    l.write('pencilCase\n')
f.close()
l.close()
image_list = open('data/image_list.txt',mode='rt',encoding='utf-8')
label_list = open('data/label_list.txt',mode='rt',encoding='utf-8')
