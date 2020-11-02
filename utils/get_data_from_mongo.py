import json
import random
import os

def db2local(save_file):
  """
  从MongoDB 获取数据，保存到save_file中
  :param save_file:
  :return:
  """
  # 配置client
  import pymongo
  client = pymongo.MongoClient("192.168.50.139", 27017)
  # 设置database
  db = client['ai-corpus']
  # 选择哪个collections
  collection = db['as_corpus']
  mydoc = collection.find({})
  with open(save_file, 'w') as f:
    for x in mydoc:
      x.pop('_id')
      content = json.dumps(x)
      f.write(content+ '\n')
  print(f"文件已生成{save_file}")

def split_all(save_file, train_rate=0.9, test_rate=0.1):
  """
  拆分成90%训练集，10%测试集
  :param save_file:
  :param train_rate: float
  :param test_rate:
  :return:
  """
  random.seed(30)
  examples = []
  with open(save_file, 'r') as f:
    lines = f.readlines()
    # 每3行一个样本
    for i in range(0, len(lines), 3):
      examples.append((lines[i],lines[i+1],lines[i+2]))
  random.shuffle(examples)
  total = len(examples)
  train_num = int(total*train_rate)
  test_num = int(total*test_rate)
  train_file = os.path.join(os.path.dirname(save_file),'train.txt')
  test_file = os.path.join(os.path.dirname(save_file),'test.txt')
  with open(train_file, 'w') as f:
    for x in examples[:train_num]:
      f.write(x[0])
      f.write(x[1])
      f.write(x[2])
  with open(test_file, 'w') as f:
    for x in examples[train_num:]:
      f.write(x[0])
      f.write(x[1])
      f.write(x[2])
  print(f"文件已生成\n {train_file}, 样本数: {train_num} \n {test_file}, 样本数: {test_num}")

def sentiment_process(save_file, new_file):
  """
  类似
  $T$ is super fast , around anywhere from 35 seconds to 1 minute .
  Boot time
  1
  :param save_file:
  :param new_file: 存储到新文件
  :return: 存储到文件
  """
  #原始文件中的sScore的映射方式
  class2id = {
    "NEG": 0,
    "NEU": 1,
    "POS":2,
  }
  id2class = {value:key for key,value in class2id.items()}
  with open(save_file, 'r') as f:
    lines = f.readlines()
  #打印多少条样本
  print_example = 10
  #总数据量
  total = 0
  with open(new_file, 'w') as f:
    for line in lines:
      line_chinese = json.loads(line)
      #使用 $T$代表apsect
      content = line_chinese["content"]
      #如果这个句子没有aspect，那就过滤掉
      if not line_chinese["aspect"]:
        continue
      for aspect in line_chinese["aspect"]:
        aspectTerm = aspect["aspectTerm"]
        sScore = aspect["sScore"]
        start = aspect["start"]
        end = aspect["end"]
        #验证一下单词的位置是否在newcontent中位置对应
        aspectTerm_insentence = "".join(content[start:end])
        if not aspectTerm == aspectTerm_insentence:
          raise Exception(f"单词在句子中位置对应不上，请检查,句子行数{total}, 句子是{line_chinese}")
        line1 = content[:start] + "$T$" + content[end:]
        line2 = aspectTerm
        # sScore映射成我们需要的, -1，0，1格式
        line3 = str(sScore-1)
        if print_example > 0:
          print(line1)
          print(line2)
          print(line3)
          print_example -= 1
        total += 1
        f.write(line1 + "\n")
        f.write(line2 + "\n")
        f.write(line3 + "\n")
  print(f"文件已生成{new_file}, 总数据量是{total}")

def check_data(save_file):
  """
  没啥用，检查下数据
  :param save_file:
  :return:
  """
  with open(save_file, 'r') as f:
    lines = f.readlines()

  without_aspect = []
  contents_lenth = []
  for line in lines:
    line_chinese = json.loads(line)
    if not line_chinese["aspect"]:
      without_aspect.append(line_chinese)
      print(line_chinese)
    else:
      contents_lenth.append(len(line_chinese["content"]))
  print(f"没有aspect的数量是{len(without_aspect)}")
  max_lenth = max(contents_lenth)
  print(f"最大的句子长度是{max_lenth}")

def clean_cache():
  """
  删除../data/cosmetics/cached* 文件
  :return:
  """
  os.system("rm -rf ../datasets/cosmetics/cached*")
  os.system("rm -rf ../logs/*")

if __name__ == '__main__':
  save_file = "../datasets/cosmetics/all.txt"
  new_file = "../datasets/cosmetics/final_all.txt"
  # db2local(save_file)
  # sentiment_process(save_file,new_file)
  # split_all(new_file,train_rate=0.9, test_rate=0.1)
  check_data(save_file)
  # clean_cache()