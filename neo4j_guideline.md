# Structure of the triplet files

## 1. The object and subject files must have the following structure

Example for Subjects (exactly the same for objects)

|index | source:ID(Subject-ID)        | :LABEL           | label  | keyword | log_level_1 | log_level_2 | log_level_3 |
|----: | :------------- |:-------------:| -----:|   -------------:  |  ------: | -----: | -------: |
| 1 | cwlb3asw....      | DIEPROFUN | DIEPROFUN |     databundel    | stuff | more stuff | even more stuff |

> the :ID is reserved by Neo4j and marks the unique ID field (used when joining triples). The (Subject-ID) is the dedicated namespace for all subject IDs.

> :LABEL is reserved by Neo4j and marks the Node type.

> label, keyword and log_level_* are attributes. We can have as many of those as we want.

## 2. The relationship file must have the following structure

|index | id | :TYPE | call_id | sentence | :START_ID(Subject-ID) |:END_ID(Object-ID) |  context_before | centext_after | 
| :------------- |:-------------:| ------------:|   ----------------:  | ----- | ----- | ------| ----------: | ---------: |
| 1 | cwea531.. | Action_Heb | asdj421... | The sentence where the triples was extracted from | The ID of Subjects (see table above) | The ID of Objects (same idea as before) | The previous sentence | The next sentence |


# Loading triplets into Neo4J


1. Open Neo4j
2. Navigate into the project 
3. Import the triplet files 
4. Open the neo4j terminal
5. Change directory and go the /bin of the project folder
6. Execute: `neo4j-admin database import full --nodes=import/subjects.csv --nodes=import/objects.csv --relationships=import/rels.csv  --overwrite-destination --skip-duplicate-nodes neo4j`
 
> `neo4j` is the name of my database instance. Adjust to your case

> `--overwrite-destination` overwrites the whole DB. It's an easy solution for our small dataset but we should not stick to it because it already bogs down 1GB of RAM for the whole I/O operation