"use client";
import { CodeHighlightTabs } from "@mantine/code-highlight";
import {
  AppShellSection,
  Box,
  Code,
  Divider,
  Group,
  Image,
  rem,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import clsx from "clsx";
import classes from "@/components/TableOfContents.module.css";
import { useEffect, useRef, useState } from "react";
import { IconListSearch } from "@tabler/icons-react";
import Link from "next/link";

const links = [
  { label: "Introduction", link: "#introduction", order: 1 },
  { label: "Overview", link: "#overview", order: 1 },
  { label: "Paraphrasing via Questioning", link: "#paraphrase", order: 1 },
  { label: "Parameters", link: "#paraphrase_parameters", order: 2 },
  { label: "Usage Example", link: "#paraphrase_usage", order: 2 },
  { label: "Summarization", link: "#summarization", order: 1 },
  { label: "Parameters", link: "#summarization_parameters", order: 2 },
  { label: "Usage Example", link: "#summarizer_usage", order: 2 },
  {
    label: "Full Example of LLM Summarizer Augmentation",
    link: "#fullExample",
    order: 1,
  },
  { label: "Full Code", link: "#fullCode", order: 2 },
  { label: "Models' Loss", link: "#modelLoss", order: 2 },
  { label: "Models' Accuracy", link: "#modelAcc", order: 2 },
];

export default function ExampleLLM() {
  const [TOCactive, setTOCactive] = useState(0);
  const sectionRefs = useRef<(HTMLDivElement | null)[]>([]); // Store references to each section

  // Set active TOC item based on scroll position
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            // Update active TOC item based on visible section
            const sectionId = entry.target.id;
            setTOCactive(
              links.findIndex((link) => link.link === `#${sectionId}`),
            );
          }
        });
      },
      { threshold: 0.6 }, // Trigger when 60% of the section is visible
    );

    // Observe each section
    sectionRefs.current.forEach((section) => {
      if (section) {
        observer.observe(section);
      }
    });

    return () => {
      observer.disconnect();
    };
  }, []);

  // Scroll to the respective section when TOC item is clicked
  const handleTOCClick = (link: string) => {
    const targetSection = document.querySelector(link);
    if (targetSection) {
      window.scrollTo({
        top: targetSection.getBoundingClientRect().top + window.scrollY - 300, // Adjust scroll position with -80px offset for the header
        behavior: "smooth",
      });
    }
  };

  const items = links.map((item, index) => (
    <Box<"a">
      component="a"
      href={item.link}
      onClick={(event) => {
        event.preventDefault();
        handleTOCClick(item.link);
      }}
      key={item.label}
      className={clsx(classes.link, {
        [classes.linkActive]: TOCactive === index,
      })}
      style={{ paddingLeft: `calc(${item.order} * var(--mantine-spacing-md))` }}
    >
      {item.label}
    </Box>
  ));
  return (
    <>
      <AppShellSection className="pt-10" bg="bg2">
        <Group justify="center">
          <Stack className="w-3/4 p-2 pb-10">
            <Title
              c="primary"
              ff="monospace"
              size={rem(42)}
              fw="bolder"
              mt="xl"
              ta="left"
            >
              Large Language Model (LLM) Augmenter
            </Title>
            <Text c="dimmed" size="lg">
              Detailed guide to using the LLM Augmenter. This page also serves
              as a proof of concept for the LLM Augmenter.
            </Text>
          </Stack>
        </Group>
        <Divider my="lg" />
      </AppShellSection>
      <AppShellSection className="flex justify-center pb-10">
        <div className="max-w-screen-lg  w-full flex space-x-8 ml-16">
          <Stack className="w-2/3">
            <Text c="primary" size="lg" fw="bolder">
              Introduction
            </Text>
            <div
              id="introduction"
              ref={(el) => {
                sectionRefs.current[0] = el;
              }}
            />
            <Text c="dimmed" size="md">
              The <Code bg="dimmed">LLMAugmenter</Code> offers an advanced
              dataset augmentation technique that leverages large language
              models (LLMs) for paraphrasing and summarization. By generating
              diverse rephrasings and summaries of input data, this augmenter
              helps prevent overfitting in text-based sentiment classification
              models by adding rich variability to the training dataset.
            </Text>
            <Text c="primary" size="lg" fw="bolder">
              Overview
            </Text>
            <div
              id="overview"
              ref={(el) => {
                sectionRefs.current[1] = el;
              }}
            />
            <Text c="dimmed" size="md">
              The <Code bg="dimmed">LLMAugmenter</Code> provides a robust
              solution to augment text data, reducing the risk of overfitting
              and enhancing model performance in NLP tasks. This augmenter
              relies on two distinct LLM-driven techniques to achieve
              variability in the dataset:
            </Text>
            <Text c="primary" size="md" className="ml-4 space-y-2">
              <Link href="/documentation/examples/llm#paraphrase">
                - Paraphrasing via Questioning
              </Link>
              <br />
              <Link href="/documentation/examples/llm#summarization">
                - Summarization
              </Link>
            </Text>

            <Divider my={2} />

            <Text c="primary" size="lg" fw="bolder">
              Paraphrasing via Questioning
            </Text>
            <div
              id="paraphrase"
              ref={(el) => {
                sectionRefs.current[2] = el;
              }}
            />
            <Text c="dimmed" size="md">
              The Paraphrasing via Questioning technique in{" "}
              <Code bg="dimmed">LLMAugmenter</Code> uses a large language model
              (LLM) to rephrase sentences by framing the task as a
              question-answering exercise. This approach prompts the model to
              reword sentences without repeating the same verbs or phrases,
              generating a unique paraphrase that maintains the original meaning
              but provides distinct wording. By framing the task as a
              question-answer prompt, the model&apos;s responses are directed
              towards producing a rephrased answer, adding both lexical and
              structural diversity.
              <br /> <br /> By generating multiple paraphrased versions of the
              same text, <Code bg="dimmed">LLMAugmenter</Code> introduces subtle
              variations that help the model learn more generalized features of
              the language. This variation reduces the chances of overfitting,
              as the model isn&apos;t exposed to identical sentences repeatedly.{" "}
              <br /> <br />
              Paraphrased sentences, with different structures and vocabulary,
              prepare the model to handle a broader range of linguistic
              patterns, improving its robustness in real-world applications.{" "}
              <br /> <br /> As the LLM avoids verb repetition while rephrasing,
              it deepens the model&apos;s understanding of summarizes and
              related terms, which is especially beneficial for tasks like
              sentiment analysis, where nuanced language is common.
            </Text>
            <Text c="primary" size="md" fw="bolder">
              Parameters
            </Text>
            <div
              id="paraphrase_parameters"
              ref={(el) => {
                sectionRefs.current[3] = el;
              }}
            />
            <Group>
              <Code bg="dimmed">sentence</Code>
              <Text>
                (str) - Input text to augment
                <br /> <i>example: &apos;This is a test sentence.&apos;</i>
              </Text>
            </Group>
            <Group>
              <Code bg="dimmed">max_new_tokens</Code>
              <Text>
                (int) - Maximum number of new tokens that can be introduced to
                the output sentence.
                <br /> <i>default: 512</i>
              </Text>
            </Group>
            <Text c="primary" size="md" fw="bolder">
              Usage Example
            </Text>
            <div
              id="paraphrase_usage"
              ref={(el) => {
                sectionRefs.current[4] = el;
              }}
            />
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: `
# Import and Initialize LLMAugmenter
from NLarge.llm import LLMAugmenter

llm_aug = LLMAugmenter()

sample_text = "This movie is a must-watch for all the family."
print(sample_text)

res = llm_aug.paraphrase_with_question(sample_text)
print(res)
`,
                  language: "python",
                },
                {
                  fileName: "output",
                  code: `
This movie is a must-watch for all the family.
This film is highly recommended for every member of the household.
  `,
                  language: "text",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />
            <Divider my={2} />

            <Text c="primary" size="lg" fw="bolder">
              Summarization
            </Text>
            <div
              id="summarization"
              ref={(el) => {
                sectionRefs.current[5] = el;
              }}
            />
            <Text c="dimmed" size="md">
              The Summarization technique in{" "}
              <Code bg="dimmed">LLMAugmenter</Code> utilizes a transformer-based
              summarizer model, specifically the BART model, to condense longer
              texts or passages into shorter, yet semantically complete
              summaries. This technique is especially useful for extracting core
              information from large documents or verbose sentences. <br />
              <br /> In NLP tasks, long sentences or paragraphs may contain
              extraneous information. Summarization helps reduce this noise by
              retaining only the most important information. <br />
              <br /> Ultimately, training on both long-form text and its
              summaries helps the model develop a nuanced understanding of
              essential versus non-essential details. This ability to
              differentiate relevant information is invaluable in tasks that
              require prioritization of critical data, like summarization,
              classification, and even information extraction.
            </Text>
            <Text c="primary" size="md" fw="bolder">
              Parameters
            </Text>
            <div
              id="summarization_parameters"
              ref={(el) => {
                sectionRefs.current[6] = el;
              }}
            />
            <Group>
              <Code bg="dimmed">text</Code>
              <Text>
                (str) - Input text to augment.
                <br /> <i>example: &apos;This is a test sentence.&apos;</i>
              </Text>
            </Group>
            <Group>
              <Code bg="dimmed">max_length</Code>
              <Text>
                (int) - Maximum length of the summary.
                <br /> <i>default: 100</i>
              </Text>
            </Group>
            <Group>
              <Code bg="dimmed">min_length</Code>
              <Text>
                (int) - Minimum length of the summary.
                <br /> <i>default: 30</i>
              </Text>
            </Group>

            <Text c="primary" size="md" fw="bolder">
              Usage Example
            </Text>
            <div
              id="summarizer_usage"
              ref={(el) => {
                sectionRefs.current[7] = el;
              }}
            />
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: `
# Import and Initialize LLMAugmenter
from NLarge.llm import LLMAugmenter

llm_aug = LLMAugmenter()

sample_text = """
Eternal Horizons is a masterpiece. It’s not just a film but an experience that lingers 
with you long after the credits roll. I highly recommend it to anyone looking for a 
deeply moving and visually enchanting cinematic experience."
"""
print(sample_text)

res = llm_aug.summarize_with_summarizer(sample_text, max_length=40, min_length=5)
print(res)
`,
                  language: "python",
                },
                {
                  fileName: "output",
                  code: `
'Eternal Horizons is a masterpiece. It’s not just a film but an experience that lingers with you long after the credits roll. I highly recommend it to anyone looking for a deeply moving and visually enchanting cinematic experience.'
'Eternal Horizons is a masterpiece.'

  `,
                  language: "text",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />
            <Divider my={2} />

            <Text c="primary" size="xl" fw="bolder">
              Full Example of LLM Summarizer Augmentation
            </Text>
            <div
              id="fullExample"
              ref={(el) => {
                sectionRefs.current[8] = el;
              }}
            />
            <Text c="dimmed" size="md">
              For your reference, below is a full example of the NLarge LLM
              Summarizer Argumentation on a dataset. This example will also
              function as a proof of concept for the NLarge LLM Summarizer
              Augmentation. This example will be evaluating augmented datasets
              on LSTM based on the loss and accuracy metrics. We have chosen the
              &apos;rotten tomatoes&apos; dataset due to it&apos;s small size
              that is prone to overfitting.
            </Text>
            <Text c="primary" size="lg" fw="bolder">
              Full Code:
            </Text>
            <div
              id="fullCode"
              ref={(el) => {
                sectionRefs.current[9] = el;
              }}
            />
            <CodeHighlightTabs
              code={[
                {
                  fileName: "python",
                  code: `
import datasets
from datasets import Dataset, Features, Value, concatenate_datasets
from NLarge.dataset_concat import augment_data, MODE
from NLarge.pipeline import TextClassificationPipeline
from NLarge.model.RNN import TextClassifierLSTM                  

original_train_data, original_test_data = datasets.load_dataset(
"rotten_tomatoes", split=["train", "test"]
)  
features = Features({"text": Value("string"), "label": Value("int64")})
original_train_data = Dataset.from_dict(
    {
        "text": original_train_data["text"],
        "label": original_train_data["label"],
    },
    features=features,
)  

# Augment and increase size by 5%, 200%
percentage= {
    MODE.LLM.SUMMARIZE: 0.05,
}
augmented_summarize_5 = augment_data(original_train_data, percentage)

percentage= {
    MODE.LLM.SUMMARIZE: 2.00,
}
augmented_summarize_200 = augment_data(original_train_data, percentage)

# Convert augmented data into Datasets
augmented_dataset_5 = Dataset.from_dict(
    {
        "text": [item["text"] for item in augmented_summarize_5],
        "label": [item["label"] for item in augmented_summarize_5],
    },
    features=features,
)
augmented_dataset_200 = Dataset.from_dict(
    {
        "text": [item["text"] for item in augmented_summarize_200],
        "label": [item["label"] for item in augmented_summarize_200],
    },
    features=features,
)

# Concatenate original and augmented datasets
augmented_train_data_5 = concatenate_datasets(
    [original_train_data, augmented_dataset_5]
)
augmented_train_data_200 = concatenate_datasets(
    [original_train_data, augmented_dataset_200]
)

# Initialize Pipelines
pipeline_augmented_5 = TextClassificationPipeline(
    augmented_data=augmented_train_data_5,
    test_data=original_test_data,
    max_length=128,
    test_size=0.2,
    model_class=TextClassifierLSTM,
)
pipeline_augmented_200 = TextClassificationPipeline(
    augmented_data=augmented_train_data_200,
    test_data=original_test_data,
    max_length=128,
    test_size=0.2,
    model_class=TextClassifierLSTM,
)

# Train Models
pipeline_augmented_5.train_model(n_epochs=10)
pipeline_augmented_200.train_model(n_epochs=10)

# Plot Loss 
pipeline_augmented_5.plot_loss(title="LSTM - LLM Summarizer Augment (5%)")
pipeline_augmented_200.plot_loss(title="LSTM - LLM Summarizer Augment (200%)")

# Plot Accuracy
pipeline_augmented_5.plot_acc(title="LSTM - LLM Summarizer Augment (5%)")
pipeline_augmented_200.plot_acc(title="LSTM - LLM Summarizer Augment (200%)")

                  `,
                  language: "python",
                },
              ]}
              className="w-full rounded-sm outline-1 outline outline-slate-600"
            />
            <Text c="primary" size="md">
              Models&apos; Loss
            </Text>
            <div
              id="modelLoss"
              ref={(el) => {
                sectionRefs.current[10] = el;
              }}
            />
            <Group justify="center">
              <Image src="/graphs/lstm_summarize_5_loss.png" />
              <Image src="/graphs/lstm_summarize_200_loss.png" />
            </Group>

            <Text c="primary" size="md">
              Models&apos; Accuracy
            </Text>
            <div
              id="modelAcc"
              ref={(el) => {
                sectionRefs.current[11] = el;
              }}
            />
            <Group justify="center">
              <Image src="/graphs/lstm_summarize_5_acc.png" />
              <Image src="/graphs/lstm_summarize_200_acc.png" />
            </Group>

            <Divider my={2} />
          </Stack>

          <div className={clsx(classes.root, "w-1/4 sticky top-8")}>
            <Group mb="md">
              <IconListSearch
                style={{ width: rem(18), height: rem(18) }}
                stroke={1.5}
              />
              <Text>Table of contents</Text>
            </Group>
            <div className={classes.links}>
              <div
                className={classes.indicator}
                style={{
                  transform: `translateY(calc(${TOCactive} * var(--link-height) + var(--indicator-offset)))`,
                }}
              />
              {items}
            </div>
          </div>
        </div>
      </AppShellSection>
    </>
  );
}
