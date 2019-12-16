import re
from .num2cn import num2cn
from .symbols import _english2latin
from .cleaners import english_cleaners
from pypinyin import pinyin, lazy_pinyin, Style
english_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

chinese_pattern = re.compile(u'[\u4e00-\u9fff]')


def char_is_chinese(c):
    return chinese_pattern.match(c)


def split_chinese_english(text):
    splitted_arr = []
    cnt = 0
    prev_cn = True
    start_cnt = 0
    for c in text:
        cur_cn = char_is_chinese(c)
        if cur_cn and not prev_cn:
            splitted_arr.append(text[start_cnt:cnt])
            start_cnt = cnt
        elif not cur_cn and prev_cn and cnt:
            splitted_arr.append(text[start_cnt:cnt])
            start_cnt = cnt
        prev_cn = cur_cn
        cnt += 1
    splitted_arr.append(text[start_cnt:cnt])
    return splitted_arr


def contains_chinese(text):
    return len(re.findall(u'[\u4e00-\u9fff]+', text)) > 0


def substr2cn(s):
    # special cases considered:
    # 1. dot
    # 3. %
    # 4. nian
    # 8. date
    # 2. @#&%*/-+=<>
    # 5. seperate english
    # 5. replace english to latin
    # 7. replace chinese symbol
    # 6. de5
    res_prefix = ''
    res_suffix = ''
    num_as_year = False
    decimal_digits = 0
    if s[-1] == '%':
        res_prefix = '百分之'
        s = s[:-1]
    if s[-1] == '年':
        res_suffix = '年'
        num_as_year = True
        s = s[:-1]

    if '.' in s:
        int_dec = s.split('.')
        if len(int_dec) == 2:
            int_part = int_dec[0]
            dec_part = int_dec[1]
            s = num2cn(int(int_part))+'点'+str2cn1by1(dec_part)
        elif len(int_dec) == 3:
            s = str2cn1by1(int_dec[0])+'点' + \
                str2cn1by1(int_dec[1])+'点'+str2cn1by1(int_dec[2])
    elif num_as_year:
        s = str2cn1by1(s)
    else:
        s = num2cn(int(s))
    return res_prefix+s+res_suffix


def str2cn1by1(s):
    res = ''
    for i in s:
        res += num2cn(int(i))
    return res


def symbol2cn(s):
    s = s.replace('@', '艾特')
    #s = s.replace('#', '井')
    #s = s.replace('%', '百分号')
    s = s.replace('+', '加')
    s = s.replace('=', '等于')
    return s


def is_ascii(s):
    return all(ord(c) < 128 for c in s) or '£' in s


def is_english(s):
    return s in (english_chars + ' ')

# TODO: check whether the start space is needed


def sep_english(s):
    # not needed function
    return s
    res = ''
    for i in range(0, len(s)):
        if i != len(s)-1:
            if is_english(s[i]) and not is_english(s[i+1]) or not is_english(s[i]) and is_english(s[i+1]):
                res += (s[i]+' ')
            else:
                res += s[i]
    res += s[-1]
    return res


def p(input):
    str = ""
    arr = pinyin(input, style=Style.TONE3)
    if len(arr) == 1 and contains_chinese(input):
        return arr[0][0]+"1 "
    for i in arr:
        str += i[0] + " "
    return str


def replace_english2latin(text):
    return text.lower().translate(text.maketrans(english_chars, _english2latin))


def replace_punc(text):
    return text.translate(text.maketrans("，。？：；！“”、（）《》—…", ",.?:;!\"\",()\"\",."))


def process_num(match):
    s = match.group()
    return substr2cn(s)


def cn_convert(s):
    if not contains_chinese(s):
        #return replace_english2latin((english_cleaners(s)))+'\n'
        return replace_punc(replace_english2latin((english_cleaners(s))))
        # return ((english_cleaners(s)))

    # num2cn
    s = re.sub('\d+\.?\d*[%年]?', process_num, s)

    return (replace_punc(replace_english2latin(symbol2cn(s))))
