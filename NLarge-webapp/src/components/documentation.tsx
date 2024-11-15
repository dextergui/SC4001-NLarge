import { Button, Divider, Text, Title } from "@mantine/core";
import { IconChevronRight } from "@tabler/icons-react";
import Link from "next/link";

type props = {
  className?: string;
  sectionsRefs: React.MutableRefObject<HTMLDivElement[]>;
};

export function Documentation({ className, sectionsRefs }: props) {
  return (
    <div className={"" + className}>
      <div
        ref={(el) => {
          if (el) {
            sectionsRefs.current[0] = el; // Assign the element to the array
          }
        }}
        id="data-augmentation"
      >
        <Title c="primary" ff="monospace" fw="bolder" my="xl" ta="left">
          Data Augmentation for Natural Language Processing
        </Title>
        <Text c="dimmed" size="lg">
          DA is a widely used technique in machine learning to enhance the
          diversity and robustness of training datasets. By artificially
          expanding the dataset, DA helps improve the generalization
          capabilities of models, particularly in scenarios where labeled data
          is scarce or expensive to obtain
          {/* Cite */}
          .In the context of Natural Language Processing (NLP), DA poses unique
          challenges due to the complexity and variability of human language.
          <br />
          <br />
          Traditional DA methods in NLP, such as synonym replacement, random
          insertion, and back-translation, have shown limited effectiveness in
          generating diverse and meaningful variations of text data. These
          methods often fail to capture the nuanced semantics and contextual
          dependencies inherent in natural language, leading to suboptimal
          improvements in model performance.
          <br />
          <br />
          Recent advancements in deep learning, particularly the development of
          Large Language Models (LLMs) like GPT-2, GPT-3, and T5, have opened
          new avenues for DA in NLP. These models, pre-trained on vast corpora
          of text data, possess a remarkable ability to generate coherent and
          contextually relevant text. Leveraging LLMs for DA involves generating
          synthetic data samples by providing prompts based on existing training
          examples.
        </Text>
        <Divider color="primary" my="lg" />
      </div>
      <div
        ref={(el) => {
          if (el) {
            sectionsRefs.current[1] = el;
          }
        }}
        id="why-does-it-matter"
      >
        <Title c="primary" ff="monospace" fw="bolder" my="xl" ta="left">
          Why does it matter?
        </Title>
        <Text c="dimmed" size="lg">
          DA has been a widely researched area in the field of Natural Language
          Processing (NLP) due to its potential to enhance the diversity and
          robustness of training datasets
          {/* \cite{DBLP:journals/corr/abs-2105-03075} */}. In the context of
          sentiment analysis, DA techniques are particularly valuable as they
          help improve the generalization capabilities of models, especially
          when labeled data is scarce
          {/* \cite{li-specia-2019-improving} */}
          .
          <br />
          <br />
          Rule based methods like random replacement are quick to implement but
          lack the generalisability to different corpus. These methods aim to
          generate new training samples by making small perturbations to the
          existing data, thereby increasing the size of the training set and
          improving the generalization capabilities of sentiment analysis models
          {/* \cite{wei-zou-2019-eda} */}
          .
          <br />
          <br />
          Interpolation methods such as synonym replacement has also been
          developed
          {/* \cite{sahin-steedman-2018-data}  */}
          where words in a sentence are replaced with their synonyms. This
          method has been shown to improve model performance by introducing
          lexical diversity. However, it often fails to capture the nuanced
          semantics and contextual dependencies inherent in natural language,
          leading to suboptimal improvements in sentiment analysis tasks
          {/* \cite{sahin-steedman-2018-data} */}
          .
          <br />
          <br />
          Leading us to the current state of the art, the use of LLMs for data
          augmentation has shown promising results in improving the performance
          of NLP models
          {/* \cite{ding-etal-2024-data} */}
          . By leveraging the generative capabilities of LLMs we are able to
          reduce the amount of noise introduced and thus generate a higher
          quality dataset. Most of the research has been focused on NER tasks,
          and we aim to explore the feasibility of using LLMs for DA in
          sentiment analysis tasks to ascertain the effectiveness of this
          approach.
          <br />
          <br />
          The benefits of LLM DA will still continue to provide superior
          performance in sentiment analysis tasks over pre-LLM DA methods.
        </Text>
        <Divider color="primary" my="lg" />
      </div>

      <div
        ref={(el) => {
          if (el) {
            sectionsRefs.current[2] = el;
          }
        }}
        id="intro-NLarge"
      >
        <Title c="primary" ff="monospace" fw="bolder" my="xl" ta="left">
          Introducing NLarge
        </Title>
        <Text c="dimmed" size="lg">
          NLarge is a Python library designed to enhance NLP model performance
          through advanced data augmentation (DA) techniques tailored for
          sentiment analysis. Our library incorporates both traditional methods
          (like random and synonym substitutions) and sophisticated techniques
          using large language models (LLMs) to generate diverse, contextually
          relevant samples. By increasing dataset variability, NLarge empowers
          models to generalize better to unseen data.
        </Text>
        <Divider color="primary" my="lg" />
      </div>

      <div
        ref={(el) => {
          if (el) {
            sectionsRefs.current[3] = el; // Assign the element to the array
          }
        }}
        id="types-of-data-aug"
      >
        <Title c="primary" ff="monospace" fw="bolder" my="xl" ta="left">
          Types of data augmentation
        </Title>
        <Text c="dimmed" size="lg">
          Our library offers three main types of data augmentation methods, each
          contributing uniquely to improved model performance:
        </Text>
      </div>
      <div
        ref={(el) => {
          if (el) {
            sectionsRefs.current[4] = el; // Assign the element to the array
          }
        }}
        id="aug-1"
      >
        <Title
          c="primary"
          ff="monospace"
          fw="bolder"
          my="xl"
          order={2}
          ta="left"
        >
          Random Substitution
        </Title>
        <Text c="dimmed" size="lg">
          Random substitution replaces words in the dataset with randomly
          selected words from the vocabulary. This technique introduces sentence
          structure variability, aiding models in learning general patterns and
          preventing overfitting.
        </Text>
        <Link href="/documentation/examples/random">
          <Button
            className="mt-4 transform transition-transform duration-300 hover:scale-105"
            c="primary"
            variant="outline"
          >
            Random Augmentation Documentation <IconChevronRight />
          </Button>
        </Link>
      </div>
      <div
        ref={(el) => {
          if (el) {
            sectionsRefs.current[5] = el; // Assign the element to the array
          }
        }}
        id="aug-2"
      >
        <Title
          c="primary"
          ff="monospace"
          fw="bolder"
          my="xl"
          order={2}
          ta="left"
        >
          Synonym Substitution
        </Title>
        <Text c="dimmed" size="lg">
          Synonym substitution swaps words for their synonyms, allowing models
          to recognize semantic similarity between different phrasings. This
          type of augmentation proved effective in creating meaningful
          variations while maintaining sentence coherence.
        </Text>
        <Link href="/documentation/examples/synonym">
          <Button
            className="mt-4 transform transition-transform duration-300 hover:scale-105"
            c="primary"
            variant="outline"
          >
            Synonym Augmentation Documentation <IconChevronRight />
          </Button>
        </Link>
      </div>
      <div
        ref={(el) => {
          if (el) {
            sectionsRefs.current[6] = el; // Assign the element to the array
          }
        }}
        id="aug-3"
      >
        <Title
          c="primary"
          ff="monospace"
          fw="bolder"
          my="xl"
          order={2}
          ta="left"
        >
          LLM-Based Augmentation
        </Title>
        <Text c="dimmed" size="lg">
          Leveraging large language models (LLMs), we employed techniques like
          paraphrasing and summarization to generate high-quality samples. These
          methods provide models with contextually diverse data, which enhances
          accuracy, particularly at extreme augmentation levels. Our studies
          revealed that summarization approaches produced fewer
          out-of-vocabulary words, further improving model performance.
        </Text>
        <Link href="/documentation/examples/llm">
          <Button
            className="mt-4 transform transition-transform duration-300 hover:scale-105"
            c="primary"
            variant="outline"
          >
            LLM Augmentation Documentation <IconChevronRight />
          </Button>
        </Link>
      </div>
      <Divider color="primary" my="lg" />

      <div
        ref={(el) => {
          if (el) {
            sectionsRefs.current[7] = el;
          }
        }}
        id="results"
      >
        <Title c="primary" ff="monospace" fw="bolder" my="xl" ta="left">
          Results and Findings
        </Title>
        <Text c="dimmed" size="lg">
          Our experiments confirmed that data augmentation, especially at higher
          levels, enhances NLP model performance for sentiment analysis. Models
          trained with 20% or more augmented data consistently outperformed
          those with lower or no augmentation. Traditional DA methods improved
          model accuracy, while LLM-based approaches offered additional
          performance gains, especially in extreme cases (200% augmentation),
          where the RNN model achieved over 90% accuracy. For researchers and
          practitioners, NLarge provides a flexible toolkit to explore, apply,
          and optimize DA strategies, helping to advance NLP model robustness
          and generalization.
        </Text>
        <Divider color="primary" my="lg" />
      </div>
    </div>
  );
}
