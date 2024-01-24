def convert_to_tokens(text, tokenizer, add_space=True):
    tokens = []
    current_token = ""
    for i in range(len(text)):
        if text[i].isspace() and set(current_token) != {" "}:
            if current_token:
                tokens.append(current_token)
                current_token = ""
            if text[i] == " ":
                current_token = " "
            else:
                tokens.append(text[i])
        else:
            current_token += text[i]
    if current_token:
        tokens.append(current_token)
    if add_space:
        if not tokens[0][0].isspace() and len(tokenizer.tokenize(tokens[0])) == len(tokenizer.tokenize(" " + tokens[0])):
            tokens[0] = " " + tokens[0]
    return tokens

ALPHABETS = "([A-Za-z])"
PREFIXES = re.compile("(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]")
SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
STARTERS = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
WEBSITES = re.compile("[.](com|net|org|io|gov|sj|bv|edu|ae)")
DIGITS = "([0-9])"
ALPHABETS_1 = re.compile("\s" + ALPHABETS + "[.] ")
ALPHABETS_2 = re.compile(ALPHABETS + "[.]" + ALPHABETS + "[.]" + ALPHABETS + "[.]")
ALPHABETS_3 = re.compile(ALPHABETS + "[.]" + ALPHABETS + "[.]")
ALPHABETS_4 = re.compile(" " + ALPHABETS + "[.]")
ACRONYMS_1 = re.compile(ACRONYMS + " " + STARTERS)
SUFFIXES_1 = re.compile(" " + SUFFIXES + "[.] " + STARTERS)
SUFFIXES_2 = re.compile(" " + SUFFIXES + "[.]")
DIGITS_1 = re.compile("[.]" + DIGITS)
ENUMERATION_1 = re.compile("( [A-Za-z0-9] )" + "[.]")
ENUMERATION_2 = re.compile("([A-Za-z0-9])" + "[.]" + "([A-Za-z0-9]+)")

def convert_to_sentences(text, tokenizer):
    "Adapted from https://stackoverflow.com/a/31505798"
    first_word = ""
    for c in text:
        if c.isspace():
            break
        first_word += c
    if text[0] in ("'", "-") or first_word and len(tokenizer.tokenize(first_word)) != len(tokenizer.tokenize("." + first_word)) - 1:
        text = " " + text
    text = re.sub(PREFIXES, "\\1<prd>", text)
    text = re.sub(WEBSITES, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(ALPHABETS_1," \\1<prd> ", text)
    text = re.sub(ACRONYMS_1, "\\1<stop> \\2", text)
    text = re.sub(ALPHABETS_2, "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(ALPHABETS_3, "\\1<prd>\\2<prd>", text)
    text = re.sub(SUFFIXES_1, " \\1<prd><stop> \\2", text)
    text = re.sub(SUFFIXES_2, " \\1<prd>", text)
    text = re.sub(ALPHABETS_4, " \\1<prd>", text)
    text = re.sub(DIGITS_1, "<prd>\\1", text)
    text = re.sub(ENUMERATION_1, "\\1<prd>", text)
    text = re.sub(ENUMERATION_2, "\\1<prd>\\2", text)
    text = text.replace(".....", "<prd><prd><prd><prd><prd>")
    text = text.replace("...", "<prd><prd><prd>")
    text = text.replace(".. ?", "<prd><prd> <qmark>")
    text = text.replace(".-", "<prd>-")
    text = text.replace("..", "<prd>.")
    text = text.replace(".@", "<prd>@")
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    if "'" in text: text = text.replace(".'", "'.")
    if ".ep" in text: text = re.sub("[.](ep \d+)( , ep \d+)*", "<prd>\\1\\2<stop>", text)
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    text = text.replace("<qmark>", "?")
    sentences = text.split("<stop>")
    for i in range(len(sentences)):
        if "”" in sentences[i]: sentences[i] = sentences[i].replace("”.", ".”")
        if "\"" in sentences[i]: sentences[i] = sentences[i].replace("\".", ".\"")
        if "!" in sentences[i]: sentences[i] = sentences[i].replace("\"!", "!\"")
        if "?" in sentences[i]: sentences[i] = sentences[i].replace("\"?", "?\"")
        if "'" in sentences[i]: sentences[i] = sentences[i].replace("'.", ".'")
    return sentences

def safe_string_get(s, i, ):
  try:
    return s[i]
  except IndexError:
    return False

def divide_into_sections(tokenized_text, tokenizer, section_length):
    if (len(tokenizer.decode(tokenized_text[section_length - 1:section_length + 2])) == 1 or
        len(tokenizer.decode(tokenized_text[section_length - 1:section_length + 2])) == 2 and
        tokenizer.decode(tokenized_text[section_length - 1:section_length + 2])[0].isspace()):
        first_part_tokens = tokenized_text[:section_length + 2]
        second_part_tokens = tokenized_text[section_length + 2:]
    elif (len(tokenizer.decode(tokenized_text[section_length - 2:section_length + 1])) == 1 or
          len(tokenizer.decode(tokenized_text[section_length - 2:section_length + 1])) == 2 and
          tokenizer.decode(tokenized_text[section_length - 2:section_length + 1])[0].isspace()):
        first_part_tokens = tokenized_text[:section_length + 1]
        second_part_tokens = tokenized_text[section_length + 1:]
    elif len(tokenizer.decode(tokenized_text[section_length - 1:section_length + 1])) == 1:
        first_part_tokens = tokenized_text[:section_length + 1]
        second_part_tokens = tokenized_text[section_length + 1:]
    else:
        first_part_tokens = tokenized_text[:section_length]
        second_part_tokens = tokenized_text[section_length:]
    return first_part_tokens, second_part_tokens

def divide_into_sections_fill(tokenized_text, tokenizer, section_length):
    index = len(tokenized_text) - section_length
    if (len(tokenizer.decode(tokenized_text[index - 1:index + 2])) == 1 or
        len(tokenizer.decode(tokenized_text[index - 1:index + 2])) == 2 and
        tokenizer.decode(tokenized_text[index - 1:index + 2])[0].isspace()):
        first_part_tokens = tokenized_text[:index + 2]
        second_part_tokens = tokenized_text[index + 2:]
    elif (len(tokenizer.decode(tokenized_text[index - 2:index + 1])) == 1 or
          len(tokenizer.decode(tokenized_text[index - 2:index + 1])) == 2 and
          tokenizer.decode(tokenized_text[index - 2:index + 1])[0].isspace()):
        first_part_tokens = tokenized_text[:index + 1]
        second_part_tokens = tokenized_text[index + 1:]
    elif len(tokenizer.decode(tokenized_text[index - 1:index + 1])) == 1:
        first_part_tokens = tokenized_text[:index + 1]
        second_part_tokens = tokenized_text[index + 1:]
    else:
        first_part_tokens = tokenized_text[:index]
        second_part_tokens = tokenized_text[index:]
    return first_part_tokens, second_part_tokens

def check_tokenization(tokenized_text, tokenized_augmented_text, tokenizer, augmentation_function, size=None):
    # if len(tokenized_augmented_text) > len(tokenized_text) and tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|endofaugmentedtext|>"))[0] in tokenized_augmented_text:
    if (augmentation_function not in PADDED_FUNCTIONS and
        augmentation_function not in REPLACE_FUNCTIONS and
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|endofaugmentedtext|>"))[0] in tokenized_augmented_text):
        # print(tokenized_augmented_text)
        # pass
        tokenized_augmented_text = tokenized_augmented_text[1:]
        # print(tokenized_text[:6])
        # if tokenized_text[:6] == [118, 94, 782, 2540, 284, 37890]:
        #     print("FIRST TOKEN: \n\n\n\n\n\n\\n\n\n\n\n\n\n\n\n\n\n\n\n", tokenized_augmented_text[0])
    if augmentation_function in PADDED_FUNCTIONS:
        padding_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|paddingtoken|>"))
        while len(tokenized_augmented_text) < len(tokenized_text):
            tokenized_augmented_text = padding_token + tokenized_augmented_text
    if augmentation_function in FILL_FUNCTIONS:
        padding_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|paddingtoken|>"))
        if len(tokenized_augmented_text) < size:
            print("TOO SHORT")
        while len(tokenized_augmented_text) < len(tokenized_text):
            tokenized_augmented_text = padding_token + tokenized_augmented_text
        if size == 1536:
            tokenized_augmented_text = tokenized_augmented_text[-1536:]
        elif size == 1024:
            tokenized_augmented_text = tokenized_augmented_text[-1024:]
        else:
            raise ValueError("invalid augmentation function")
        return tokenized_augmented_text
    if augmentation_function in REPLACE_FUNCTIONS:
        debug = False
        if tokenized_text[0] == 8244:
            debug = True
        # print(len(tokenized_text))
        # print(len(tokenized_augmented_text))
        # print(tokenized_text)
        # print(tokenized_augmented_text)
        # assert False
        replacement_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|replacement|>"))
        end_of_augmented_text_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|endofaugmentedtext|>"))[0]
        i, j = 0, 0
        while i < len(tokenized_text) and j < len(tokenized_text):
            # if debug:
            #     print(tokenized_augmented_text[j], tokenized_text[i])
            if tokenized_augmented_text[j] == end_of_augmented_text_token and tokenizer.decode(tokenized_augmented_text[j+1]).strip() == tokenizer.decode(tokenized_text[i]).strip() and j >= 512:
                if tokenizer.decode(tokenized_augmented_text[j+1]).strip() == "":
                    if tokenized_augmented_text[j+1] == tokenized_text[i]:
                        break
                else:
                    break
                # tokenized_augmented_text = tokenized_augmented_text[:j] + replacement_token + tokenized_augmented_text[j:]
                # i += 1
                # j += 1
                # continue
            if tokenizer.decode(tokenized_text[i]).strip() != tokenizer.decode(tokenized_augmented_text[j]).strip():
                tokenized_augmented_text = tokenized_augmented_text[:j] + replacement_token + tokenized_augmented_text[j:]
            i += 1
            j += 1
        tokenized_augmented_text = tokenized_augmented_text[1:]
    # print(len(tokenized_text))
    if len(tokenized_augmented_text) != len(tokenized_text):
        # print(find_difference(tokenized_text, tokenized_augmented_text))
        # pass
        tokenized_augmented_text = fix_tokenization(tokenized_text, tokenized_augmented_text, tokenizer, augmentation_function)
    if len(tokenized_text) == len(tokenized_augmented_text):
        # print(len(tokenized_text))
        # print(len(tokenized_augmented_text))
        # print(tokenized_text)
        # print(tokenized_augmented_text)
        # assert False
        return tokenized_augmented_text
    else:
        return tokenized_augmented_text
        # print(len(tokenized_text))
        # print(len(tokenized_augmented_text))
        # print(tokenized_text)
        # print(tokenized_augmented_text)
        # assert False
        # return False

def fix_tokenization(tokenized_text, tokenized_augmented_text, tokenizer, augmentation_function):
    if len(tokenized_text) > len(tokenized_augmented_text):
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 2343 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [2343, 226] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 16268 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [16268, 249] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 19567 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 19567] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 10545 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [10545, 246] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 136 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 136] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 28053 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [28053, 120] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 133 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 133] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 156 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 156] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 27332 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [27332, 119] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 132 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 132] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 20015 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 20015] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 10263 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [10263, 227] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 134 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 134] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 130 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 130] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 27670 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 27670] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 5099 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 5099] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 26292 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 26292] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 142 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 142] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 157 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [157, 118] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 156 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [156, 106] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 34247 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [220, 34247] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 161 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [161, 254] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 165 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [165, 253] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 162 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [162, 249] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 115 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [115, 253] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 163 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [163, 114] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 169 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [169, 247] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and 164 in tokenized_text:
            index = tokenized_augmented_text.index(4210)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [164, 111] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 98 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [98, 232] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 119 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [119, 229] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 224 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [224, 117] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 223 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [223, 226] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 99 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [99, 236] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 235 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [235, 119] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 115 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [115, 243] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 120 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [120, 242] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 252 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [252, 234] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 122 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [122, 110] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 253 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [253, 111] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 102 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [102, 109] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 226 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [226, 244] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 118 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [118, 96] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 225 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [225, 254] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 233 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [233, 227] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 123 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [123, 114] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 247 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [247, 101] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 255 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [255, 96] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and 112 in tokenized_text:
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [112, 108] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(1828) > tokenized_text.count(1828) and 17 in tokenized_text:
            index = tokenized_augmented_text.index(1828)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [17, 17] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(10332) > tokenized_text.count(10332) and 519 in tokenized_text:
            index = tokenized_augmented_text.index(10332)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [519, 70] + tokenized_augmented_text[index + 1:]
        if tokenized_augmented_text.count(6353) > tokenized_text.count(6353) and tokenizer.decode(tokenized_text[:2]) == tokenizer.decode(6353):
            index = tokenized_augmented_text.index(6353)
            tokenized_augmented_text = tokenized_augmented_text[:index] + tokenized_text[:2] + tokenized_augmented_text[index + 1:]
        if tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and tokenized_augmented_text[-1] == 20543:
            tokenized_augmented_text = tokenized_augmented_text[:-1] + tokenized_text[-2:]
        if tokenized_augmented_text.count(40670) > tokenized_text.count(40670) and tokenized_augmented_text[-1] == 40670:
            tokenized_augmented_text = tokenized_augmented_text[:-1] + tokenized_text[-2:]
        if tokenized_augmented_text.count(4210) > tokenized_text.count(4210) and tokenized_augmented_text[-1] == 4210:
            tokenized_augmented_text = tokenized_augmented_text[:-1] + tokenized_text[-2:]
        if tokenized_text.count(107) > tokenized_augmented_text.count(107) and tokenized_augmented_text[-1] == 156:
            tokenized_augmented_text = tokenized_augmented_text + [107]
        if tokenized_text.count(113) > tokenized_augmented_text.count(113) and tokenized_augmented_text[-1] == 156:
            tokenized_augmented_text = tokenized_augmented_text + [113]
        if tokenized_augmented_text[-1] == 156 and tokenized_text[-2] == 156:
            tokenized_augmented_text = tokenized_augmented_text + tokenized_text[-1:]
        if tokenized_augmented_text.count(48585) > tokenized_text.count(48585) and tokenizer.decode(tokenized_text[:3]) == tokenizer.decode(48585):
            index = tokenized_augmented_text.index(48585)
            tokenized_augmented_text = tokenized_augmented_text[:index] + tokenized_text[:3] + tokenized_augmented_text[index + 1:]
        if tokenized_text.count(13) > tokenized_augmented_text.count(13) and tokenized_text[-3:] == [13, 163, 106]:
            tokenized_augmented_text = tokenized_augmented_text[:-2] + [13, 163, 106]
    if len(tokenized_augmented_text) > len(tokenized_text):
        while tokenized_text.count(34247) > tokenized_augmented_text.count(34247) and 12919 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(12919)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [34247] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(27032) > tokenized_augmented_text.count(27032) and 5641 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(5641)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [27032] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(15474) > tokenized_augmented_text.count(15474) and 5641 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(5641)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [15474] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(12045) > tokenized_augmented_text.count(12045) and 6312 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(6312)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [12045] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(127) > tokenized_augmented_text.count(127) and 157 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(157)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [127] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(33951) > tokenized_augmented_text.count(33951) and 25529 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(25529)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [33951] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(25443) > tokenized_augmented_text.count(25443) and 15166 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(15166)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [25443] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(234) > tokenized_augmented_text.count(234) and 10263 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(10263)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [234] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(2515) > tokenized_augmented_text.count(2515) and 163 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(163)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [2515] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(35050) > tokenized_augmented_text.count(35050) and 114 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(114)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [35050] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(17683) > tokenized_augmented_text.count(17683) and 5641 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(5641)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [17683] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(49149) > tokenized_augmented_text.count(49149) and 5641 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(5641)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [49149] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(35050) > tokenized_augmented_text.count(35050) and 20543 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [35050] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(118) < tokenized_augmented_text.count(118) and 157 in tokenized_augmented_text and 157 in tokenized_text:
            index = tokenized_augmented_text.index(118)
            tokenized_augmented_text = tokenized_augmented_text[:index] + tokenized_augmented_text[index + 1:]
        while tokenized_text.count(103) > tokenized_augmented_text.count(103) and 16268 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(16268)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [103] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(224) > tokenized_augmented_text.count(224) and 16268 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(16268)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [224] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(116) > tokenized_augmented_text.count(116) and 161 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(161)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [116] + tokenized_augmented_text[index + 2:]
        while tokenized_augmented_text.count(2238) > tokenized_text.count(2238) and 5488 in tokenized_text and 2364 in tokenized_text:
            index = tokenized_augmented_text.index(2238)
            tokenized_augmented_text = tokenized_augmented_text[:index-1] + [5488, 572] + tokenized_augmented_text[index + 2:]
        while tokenized_augmented_text.count(297) > tokenized_text.count(297) and 1183 in tokenized_text:
            index = tokenized_augmented_text.index(297)
            tokenized_augmented_text = tokenized_augmented_text[:index] + tokenized_augmented_text[index + 1:]
        # while tokenized_augmented_text.count(11934) > tokenized_text.count(11934) and 23129 in tokenized_text and 32775 in tokenized_text:
        #     index = tokenized_augmented_text.index(11934)
        #     tokenized_augmented_text = tokenized_augmented_text[:index-2] + [23129, 764] + tokenized_augmented_text[index + 2:]
        while tokenized_augmented_text.count(3359) > tokenized_text.count(3359) and 11298 in tokenized_text and 889 in tokenized_text:
            index = tokenized_augmented_text.index(3359)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [11298, 889] + tokenized_augmented_text[index + 3:]
        # while tokenized_augmented_text.count(1158) > tokenized_text.count(1158) and 1083 in tokenized_text and 607 in tokenized_text:
        #     index = tokenized_augmented_text.index(1158)
        #     tokenized_augmented_text = tokenized_augmented_text[:index-2] + [607, 1083] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(5872) > tokenized_text.count(5872) and 12453 in tokenized_text and 14864 in tokenized_text:
            index = tokenized_augmented_text.index(5872)
            tokenized_augmented_text = tokenized_augmented_text[:index-1] + [14864, 12453] + tokenized_augmented_text[index + 2:]
        while tokenized_augmented_text.count(6519) > tokenized_text.count(6519) and 12048 in tokenized_text and 896 in tokenized_text:
            index = tokenized_augmented_text.index(6519)
            tokenized_augmented_text = tokenized_augmented_text[:index-2] + [896, 12048] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(949) > tokenized_text.count(949) and 5664 in tokenized_text and 358 in tokenized_text:
            index = tokenized_augmented_text.index(949)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [5664, 358] + tokenized_augmented_text[index + 3:]
        while tokenized_augmented_text.count(520) > tokenized_text.count(520) and 7402 in tokenized_text and 17738 in tokenized_text:
            index = tokenized_augmented_text.index(520)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [17738, 7402] + tokenized_augmented_text[index + 4:]
        while tokenized_augmented_text.count(14014) > tokenized_text.count(14014) and 3776 in tokenized_text and 5558 in tokenized_text:
            index = tokenized_augmented_text.index(14014)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [3776, 5558] + tokenized_augmented_text[index + 3:]
        while tokenized_augmented_text.count(667) > tokenized_text.count(667) and 20282 in tokenized_text and 3701 in tokenized_text:
            index = tokenized_augmented_text.index(667)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [3701, 20282] + tokenized_augmented_text[index + 3:]
        while tokenized_augmented_text.count(359) > tokenized_text.count(359) and 2171 in tokenized_text and 577 in tokenized_text:
            index = tokenized_augmented_text.index(359)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [2171, 577] + tokenized_augmented_text[index + 3:]
        while tokenized_augmented_text.count(255) > tokenized_text.count(255) and 29690 in tokenized_text:
            index = tokenized_augmented_text.index(255)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [29690] + tokenized_augmented_text[index + 2:]
        while tokenized_augmented_text.count(488) > tokenized_text.count(488) and 1872 in tokenized_text:
            index = tokenized_augmented_text.index(488)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [16590, 1872] + tokenized_augmented_text[index + 3:]
        while tokenized_augmented_text.count(4782) > tokenized_text.count(4782) and 5620 in tokenized_text:
            index = tokenized_augmented_text.index(4782)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [5620] + tokenized_augmented_text[index + 2:]
        while tokenized_augmented_text.count(3897) > tokenized_text.count(3897) and 79 in tokenized_text:
            index = tokenized_augmented_text.index(3897)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [79, 2913, 7791] + tokenized_augmented_text[index + 4:]
        while tokenized_augmented_text.count(230) > tokenized_text.count(230) and 42062 in tokenized_text:
            index = tokenized_augmented_text.index(230)
            if tokenized_augmented_text[index + 1] == 4210:
                tokenized_augmented_text = tokenized_augmented_text[:index] + [42062] + tokenized_augmented_text[index + 2:]
            else:
                break
        while tokenized_augmented_text.count(20543) > tokenized_text.count(20543) and 42062 in tokenized_text:
            index = tokenized_augmented_text.index(20543)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [42062] + tokenized_augmented_text[index + 2:]
        while tokenized_augmented_text.count(67) > tokenized_text.count(67) and 1549 in tokenized_text:
            index = tokenized_augmented_text.index(67)
            tokenized_augmented_text = tokenized_augmented_text[:index-1] + [1549] + tokenized_augmented_text[index + 1:]
        while tokenized_augmented_text.count(28053) > tokenized_text.count(28053) and 247 in tokenized_text:
            index = tokenized_augmented_text.index(28053)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [247] + tokenized_augmented_text[index + 2:]
        while tokenized_augmented_text.count(28053) > tokenized_text.count(28053) and 564 in tokenized_text:
            index = tokenized_augmented_text.index(28053)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [564] + tokenized_augmented_text[index + 2:]
        while tokenized_augmented_text.count(5525) < tokenized_text.count(5525) and 16268 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(16268)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [5525] + tokenized_augmented_text[index + 2:]
        while tokenized_augmented_text.count(164) < tokenized_text.count(164) and 16268 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(16268)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [164] + tokenized_augmented_text[index + 2:]
        while tokenized_augmented_text.count(245) < tokenized_text.count(245) and 16268 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(16268)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [245] + tokenized_augmented_text[index + 2:]
        while tokenized_augmented_text.count(98) > tokenized_text.count(98) and 94 in tokenized_text:
            index = tokenized_augmented_text.index(98)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [94] + tokenized_augmented_text[index + 2:]
        while tokenized_augmented_text.count(102) < tokenized_text.count(102) and 16268 in tokenized_augmented_text:
            index = tokenized_augmented_text.index(16268)
            tokenized_augmented_text = tokenized_augmented_text[:index] + [102] + tokenized_augmented_text[index + 2:]
        while tokenized_text.count(226) < tokenized_augmented_text.count(226) and 5323 in tokenized_augmented_text and 5323 in tokenized_text:
            index = tokenized_augmented_text.index(5323)
            if tokenized_augmented_text[index - 1] == 226:
                tokenized_augmented_text = tokenized_augmented_text[:index - 1] + tokenized_augmented_text[index:]
            else:
                break
        while tokenized_text.count(226) < tokenized_augmented_text.count(226) and 2343 in tokenized_augmented_text and 2343 in tokenized_text:
            index = tokenized_augmented_text.index(2343)
            if tokenized_augmented_text[index + 1] == 226:
                tokenized_augmented_text = tokenized_augmented_text[:index + 1] + tokenized_augmented_text[index + 2:]
            else:
                break
        while tokenized_text.count(220) < tokenized_augmented_text.count(220) and 156 in tokenized_augmented_text and 156 in tokenized_text:
            index = tokenized_augmented_text.index(156)
            if tokenized_augmented_text[index - 1] == 220:
                tokenized_augmented_text = tokenized_augmented_text[:index - 1] + tokenized_augmented_text[index:]
            else:
                break
        while tokenized_text.count(297) < tokenized_augmented_text.count(297) and 297 in tokenized_augmented_text and 1183 in tokenized_text:
            index = tokenized_augmented_text.index(297)
            if tokenized_augmented_text[index - 1] == 705:
                tokenized_augmented_text = tokenized_augmented_text[:index - 1] + [1183] + tokenized_augmented_text[index + 1:]
            else:
                break
        if tokenized_augmented_text[-2:] == [157, 118] and tokenized_text[-1] == 157:
            tokenized_augmented_text = tokenized_augmented_text[:-1]
        if len(tokenized_augmented_text) > len(tokenized_text) and augmentation_function == "shuffle_within_sentences_high_pmi":
            diff = find_difference(tokenized_text, tokenized_augmented_text)
            if tokenizer.decode(diff["in second"]) == tokenizer.decode(diff["in first"][::-1]) and len(diff["in second"]) > 0:
                index = tokenized_augmented_text.index(diff["in second"][0])
                tokenized_augmented_text = tokenized_augmented_text[:index] + diff["in first"][::-1] + tokenized_augmented_text[index + len(diff["in second"]):]
    if len(tokenized_text) > len(tokenized_augmented_text):
        if tokenizer.decode(tokenized_text[:2]) == tokenizer.decode(6353) and 6353 not in tokenized_augmented_text:
            tokenized_augmented_text = tokenized_text[1:2] + tokenized_augmented_text
    return tokenized_augmented_text