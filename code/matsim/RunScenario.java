import java.util.*;
import org.matsim.api.core.v01.Scenario;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.groups.*;
import org.matsim.core.controler.Controler;
import org.matsim.core.controler.OutputDirectoryHierarchy;
import org.matsim.core.scenario.ScenarioUtils;

/**
 * Runs a MATSim scenario built from the tours-pipeline export.
 *
 * Args:
 *   [0] network.xml.gz
 *   [1] plans.xml            (from matsim_export.py)
 *   [2] transitSchedule.xml.gz
 *   [3] transitVehicles.xml.gz
 *   [4] outputDir
 *   [5] sampleFrac           (population sample fraction; scales flow/storage capacity)
 *   [6] lastIteration        (e.g. 10)
 *   [7] endTimeHours         (mobsim end, e.g. 30)
 *   [8] "modeswitch"         optional: enable ChangeExpBeta mode switching
 *                            (default OFF — see replanning note below)
 *   [9] innovFrac            optional: fractionOfIterationsToDisableInnovation
 *                            (default 0.8; DynusT-like equilibration tail with
 *                            pure plan selection and no innovation)
 *  [10] rerouteSched         optional: "w0,w1,w2" ReRoute weights for the
 *                            0-30% / 0-60% / 0-innovFrac iteration bands
 *                            (default "0.3,0.2,0.1")
 *
 * Car is routed on the network, pt uses the GTFS transit schedule, and
 * walk/bike/ride are teleported. When sampleFrac < 1 the flow and storage
 * capacity factors are set to sampleFrac (the standard MATSim scaling so a
 * p% population sample loads the network like the full population).
 *
 * DynusT behaviour: DynusT is a simulation-based dynamic traffic assignment —
 * fixed modes, iterative route choice with averaging, flows stabilising to
 * equilibrium. This driver mirrors that with a ReRoute-ONLY innovation
 * schedule whose weight steps down (w0 -> w0+w1 -> ... as bands disable) and
 * an innovation-free tail after innovFrac of the iterations, so the
 * iteration-to-iteration link-volume gap (see matsim_analysis.py) can settle
 * the way DynusT route flows do. No ChangeExpBeta by default (modes stay as
 * the activity plans specify, as in a DynusT DTA run).
 */
public class RunScenario {
    public static void main(String[] args) {
        String network   = args[0];
        String plans     = args[1];
        String schedule  = args[2];
        String vehicles  = args[3];
        String outputDir = args[4];
        double sampleFrac = args.length > 5 ? Double.parseDouble(args[5]) : 1.0;
        int lastIter      = args.length > 6 ? Integer.parseInt(args[6]) : 5;
        double endHours   = args.length > 7 ? Double.parseDouble(args[7]) : 30.0;
        double capFactor  = Math.max(0.001, Math.min(1.0, sampleFrac));

        Config config = ConfigUtils.createConfig();
        config.network().setInputFile(network);
        config.plans().setInputFile(plans);
        config.global().setCoordinateSystem("EPSG:28356");

        config.controller().setOutputDirectory(outputDir);
        config.controller().setFirstIteration(0);
        config.controller().setLastIteration(lastIter);
        config.controller().setOverwriteFileSetting(
            OutputDirectoryHierarchy.OverwriteFileSetting.deleteDirectoryIfExists);
        config.controller().setWriteEventsInterval(Math.max(1, lastIter));
        config.controller().setWritePlansInterval(Math.max(1, lastIter));

        // public transit (from GTFS)
        config.transit().setUseTransit(true);
        config.transit().setTransitScheduleFile(schedule);
        config.transit().setVehiclesFile(vehicles);
        config.transit().setTransitModes(new HashSet<>(Arrays.asList("pt")));

        // qsim + capacity scaling for the population sample
        config.qsim().setStartTime(0);
        config.qsim().setEndTime(endHours * 3600);
        config.qsim().setMainModes(Arrays.asList("car"));
        config.qsim().setFlowCapFactor(capFactor);
        config.qsim().setStorageCapFactor(capFactor);
        config.qsim().setNumberOfThreads(Math.max(1,
            Runtime.getRuntime().availableProcessors() - 1));

        // link statistics: written EVERY iteration (single-iteration average)
        // so the analysis can measure iteration-to-iteration volume change —
        // the DTA-primer's convergence criterion (relative gap). Without this
        // only one end-of-run average exists and convergence is unverifiable.
        config.linkStats().setWriteLinkStatsInterval(1);
        config.linkStats().setAverageLinkStatsOverIterations(1);

        // routing: car on network, others teleported
        RoutingConfigGroup routing = config.routing();
        routing.setNetworkModes(Arrays.asList("car"));
        addTele(routing, "walk", 1.2, 1.3);
        addTele(routing, "bike", 4.0, 1.4);
        addTele(routing, "ride", 12.0, 1.3);

        // scoring: activity types present in the tours-pipeline plans
        ScoringConfigGroup scoring = config.scoring();
        addAct(scoring, "home", 12 * 3600);
        addAct(scoring, "work", 8 * 3600);
        addAct(scoring, "edu", 6 * 3600);
        addAct(scoring, "shopping", 3600);
        addAct(scoring, "escort", 1800);
        addAct(scoring, "social", 7200);
        addAct(scoring, "othmaint", 3600);
        addAct(scoring, "eat", 1800);
        addAct(scoring, "othdiscr", 3600);
        addAct(scoring, "other", 3600);
        for (String m : new String[]{"car", "pt", "walk", "bike", "ride"}) {
            scoring.addModeParams(new ScoringConfigGroup.ModeParams(m));
        }

        // replanning. Mode switching (ChangeExpBeta) is OFF by default: with
        // all mode constants at zero, uncongested teleported ride/bike
        // outscore network car and the population abandons car within a few
        // iterations (observed: 148k car legs in plans -> 19.7k car departures
        // by it.10). Pass "modeswitch" as arg [8] to enable it.
        // DynusT-like schedule: ReRoute ONLY (fixed modes, as in DTA), weight
        // stepping down across iteration bands, then an innovation-free tail
        // (fractionOfIterationsToDisableInnovation) so route flows stabilise.
        // Overlapping ReRoute entries SUM while simultaneously active, giving
        // total reroute probability w0+w1+w2 early, w1+w2 mid, w2 late, 0 tail.
        ReplanningConfigGroup repl = config.replanning();
        boolean modeSwitch = args.length > 8 && "modeswitch".equalsIgnoreCase(args[8]);
        double innovFrac = args.length > 9 ? parseInnovFrac(args[9]) : 0.8;
        double[] rrW = args.length > 10 ? parseSchedule(args[10]) : new double[]{0.3, 0.2, 0.1};
        repl.setFractionOfIterationsToDisableInnovation(innovFrac);
        ReplanningConfigGroup.StrategySettings sel = new ReplanningConfigGroup.StrategySettings();
        sel.setStrategyName("ChangeExpBeta");
        sel.setWeight(modeSwitch ? 0.7 : 0.0);
        repl.addStrategySettings(sel);
        int b1 = bandEnd(lastIter, 0.3, 1);
        int b2 = bandEnd(lastIter, 0.6, b1);
        int b3 = bandEnd(lastIter, innovFrac, b2);
        addReroute(repl, rrW[0], b1);
        addReroute(repl, rrW[1], b2);
        addReroute(repl, rrW[2], b3);

        Scenario scenario = ScenarioUtils.loadScenario(config);
        new Controler(scenario).run();
    }

    private static void addReroute(ReplanningConfigGroup repl, double weight, int disableAfter) {
        ReplanningConfigGroup.StrategySettings rr = new ReplanningConfigGroup.StrategySettings();
        rr.setStrategyName("ReRoute");
        rr.setWeight(weight);
        rr.setDisableAfterIteration(disableAfter);
        repl.addStrategySettings(rr);
    }

    private static int bandEnd(int lastIter, double frac, int atLeast) {
        int b = (int) Math.round(lastIter * frac);
        if (b < atLeast) b = atLeast;
        if (b > lastIter) b = lastIter;
        return b;
    }

    private static double parseInnovFrac(String s) {
        try {
            double v = Double.parseDouble(s.trim());
            if (v < 0.5) return 0.5;
            if (v > 1.0) return 1.0;
            return v;
        } catch (NumberFormatException e) {
            return 0.8;
        }
    }

    private static double[] parseSchedule(String s) {
        double[] dflt = new double[]{0.3, 0.2, 0.1};
        try {
            String[] parts = s.trim().split(",");
            if (parts.length != 3) return dflt;
            double[] w = new double[3];
            for (int i = 0; i < 3; i++) {
                w[i] = Double.parseDouble(parts[i].trim());
                if (w[i] < 0.0 || w[i] > 1.0) return dflt;
            }
            return w;
        } catch (NumberFormatException e) {
            return dflt;
        }
    }

    private static void addTele(RoutingConfigGroup routing, String mode,
                                double speed, double beeline) {
        RoutingConfigGroup.TeleportedModeParams p =
            new RoutingConfigGroup.TeleportedModeParams(mode);
        p.setTeleportedModeSpeed(speed);
        p.setBeelineDistanceFactor(beeline);
        routing.addTeleportedModeParams(p);
    }

    private static void addAct(ScoringConfigGroup scoring, String type, double dur) {
        ScoringConfigGroup.ActivityParams p = new ScoringConfigGroup.ActivityParams(type);
        p.setTypicalDuration(dur);
        scoring.addActivityParams(p);
    }
}
