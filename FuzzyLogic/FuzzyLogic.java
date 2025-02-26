import net.sourceforge.jFuzzyLogic.FIS;

public class FuzzyLogic {
    public static void main(String[] args) {
        // Load and parse the FCL file that defines fuzzy sets, rules, and defuzzification.
        FIS fis = FIS.load("./FuzzyLogic/fcl/funding.fcl", true);

        // This value will be fuzzified according to the membership functions defined in the FCL.
        fis.setVariable("funding", 60);

        // Like funding, this value is fuzzified into its corresponding fuzzy sets.
        fis.setVariable("staffing", 14);

        // This step applies the fuzzification, rule evaluation, aggregation,
        // and defuzzification (using the center of gravity method) processes.
        fis.evaluate();

        //fis.chart();

        // The printed value (approximately 37.284774495405784) is computed by:
        //  1. Fuzzifying the inputs (mapping them to their membership values).
        //  2. Evaluating the fuzzy rules (e.g. funding and staffing affecting risk).
        //  3. Aggregating the output fuzzy sets based on the rules' firing strengths.
        //  4. Finally defuzzifying the aggregated set using the center-of-gravity method.
        System.out.println(fis.getVariable("risk").getValue());
    }
}