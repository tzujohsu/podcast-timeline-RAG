from tqdm import tqdm
from zhpr.predict import DocumentDataset,merge_stride,decode_pred
from transformers import AutoModelForTokenClassification,AutoTokenizer
from torch.utils.data import DataLoader

# configuration
window_size = 256
step = 200
model_name = 'p208p2002/zh-wiki-punctuation-restore'
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def predict_step(batch,model, tokenizer):
        batch_out = []
        batch_input_ids = batch

        encodings = {'input_ids': batch_input_ids}
        output = model(**encodings)

        predicted_token_class_id_batch = output['logits'].argmax(-1)
        for predicted_token_class_ids, input_ids in zip(predicted_token_class_id_batch, batch_input_ids):
            out=[]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)

            # compute the pad start in input_ids
            # and also truncate the predict
            # print(tokenizer.decode(batch_input_ids))
            input_ids = input_ids.tolist()
            try:
                input_id_pad_start = input_ids.index(tokenizer.pad_token_id)
            except:
                input_id_pad_start = len(input_ids)
            input_ids = input_ids[:input_id_pad_start]
            tokens = tokens[:input_id_pad_start]

            # predicted_token_class_ids
            predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids]
            predicted_tokens_classes = predicted_tokens_classes[:input_id_pad_start]

            for token,ner in zip(tokens,predicted_tokens_classes):
                out.append((token,ner))
            batch_out.append(out)
        return batch_out

import re

def replace_english_with_placeholders(text):
    """
    Replaces English text in a mixed Chinese-English input with numbered placeholders.

    Args:
        text (str): Input text containing both Chinese and English.

    Returns:
        tuple: Processed text with placeholders and a mapping of placeholders to English text.
    """
    # Find all English segments
    english_texts = re.findall(r'[a-zA-Z0-9\\s]+', text)
    placeholder_mapping = {}

    # Replace in one pass
    def replacer(match):
        nonlocal placeholder_mapping
        eng_text = match.group(0)
        placeholder = f"<{len(placeholder_mapping) + 1}>"
        placeholder_mapping[placeholder] = eng_text.strip()
        return placeholder

    processed_text = re.sub(r'[a-zA-Z0-9\\s]+', replacer, text)

    return processed_text, placeholder_mapping


def restore_english_from_placeholders(processed_text, placeholder_mapping):
    """
    Restores English text in a processed text by replacing placeholders with original English strings.

    Args:
        processed_text (str): Text with numbered placeholders.
        placeholder_mapping (dict): Mapping of placeholders to English text.

    Returns:
        str: Text with placeholders replaced by the original English text.
    """
    for placeholder, eng_text in placeholder_mapping.items():
        processed_text = processed_text.replace(placeholder, eng_text)

    return processed_text

def replace_unk_with_space(text):
    return re.sub(r"\[UNK]", " ", text)


def restore_punctuation(text):
    processed_text, mapping = replace_english_with_placeholders(text)

    dataset = DocumentDataset(processed_text, window_size=window_size,step=step)
    dataloader = DataLoader(dataset=dataset,shuffle=False, batch_size=10)
    model_pred_out = []
    for batch in tqdm(dataloader):
        batch_out = predict_step(batch,model,tokenizer)
        for out in batch_out:
            model_pred_out.append(out)

    merge_pred_result = merge_stride(model_pred_out,step)
    merge_pred_result_deocde = decode_pred(merge_pred_result)
    merge_pred_result_deocde = ''.join(merge_pred_result_deocde)
    restored_text = restore_english_from_placeholders(
        merge_pred_result_deocde,
        mapping
    )
    restored_text = replace_unk_with_space(restored_text)
    return restored_text
    